import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ---- Config Paths ---- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Diffusion Process ---- #
class Diffusion:
    def __init__(self, device, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_cumprod_t = self.alpha_cumprod[t.long()].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)


# ---- Attention Modules ---- #
class CrossAttention2D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, y):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H * W).permute(0, 2, 1)

        # TODO: This interpolation is used to prevent shape mismatch during attention,
        #       but may introduce artifacts. Consider replacing with learned upsampling later.
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

        # Ensure shape matches after interpolation to avoid reshape errors
        if y.shape[2] != H or y.shape[3] != W:
            raise ValueError(f"Expected interpolated y to have shape ({B}, {C}, {H}, {W}), but got {y.shape}")
        try:
            y_reshaped = y.view(B, C, H * W).permute(0, 2, 1)
        except RuntimeError as e:
            print(f"[Error Reshaping y] B={B}, C={C}, H={H}, W={W}, y.shape={y.shape}")
            raise RuntimeError("Failed to reshape conditioning tensor 'y' for attention. Check channel count and spatial dimensions.") from e
        attn_out, _ = self.mha(x_reshaped, y_reshaped, y_reshaped)
        x_reshaped = self.activation(x_reshaped + attn_out)
        return x_reshaped.permute(0, 2, 1).view(B, C, H, W)


class PaCA(nn.Module):
    def __init__(self, in_channels, cond_channels, heads=4):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(cond_channels, in_channels, 1)
        self.value = nn.Conv2d(cond_channels, in_channels, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        self.heads = heads

    def forward(self, x, cond):
        B, C, H, W = x.size()
        q = self.query(x).reshape(B, self.heads, C // self.heads, H * W)
        k = self.key(cond).reshape(B, self.heads, C // self.heads, H * W)
        v = self.value(cond).reshape(B, self.heads, C // self.heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / (C ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        return self.out(out) + x


# ---- Flexible Attention Wrapper ---- #
def get_attention_module(attn_type, channels, cond_channels=None, heads=4):
    if attn_type == 'mha':
        return CrossAttention2D(channels, heads)
    elif attn_type == 'paca':
        return PaCA(channels, cond_channels or channels, heads)
    else:
        raise ValueError("Unknown attention type: choose 'mha' or 'paca'")


# ---- ControlNet-style Conditioning ---- #
class ControlNetLDM(nn.Module):
    def __init__(self, base_unet, control_channels, attn_type='mha', num_heads=4):
        super().__init__()
        self.control_attn = get_attention_module(attn_type, channels=4, cond_channels=control_channels, heads=num_heads)

        self.down_blocks = base_unet.down_blocks
        self.mid_block = base_unet.mid_block
        self.up_blocks = base_unet.up_blocks
        self.conv_in = base_unet.conv_in
        self.conv_norm_out = base_unet.conv_norm_out
        self.conv_act = base_unet.conv_act
        self.conv_out = base_unet.conv_out

        self.attn_blocks = nn.ModuleList([
            get_attention_module(attn_type, channels=320, cond_channels=control_channels, heads=num_heads)
            for _ in self.up_blocks
        ])

    def forward(self, x, t, cbct_cond):
        # Create dummy encoder_hidden_states because UNet expects conditioning from text
        null_emb = torch.zeros((x.size(0), 77, 768), device=x.device)
        x = self.control_attn(x, cbct_cond)
        emb = base_unet.time_proj(t)
        emb = base_unet.time_embedding(emb)
        residuals = []
        x = self.conv_in(x)
        for block in self.down_blocks:
            out = block(x, emb, null_emb)
            if isinstance(out, tuple) and len(out) == 2:
                x, res = out
            else:
                x, res = out, None
            residuals.append(res)

        x = self.mid_block(x, emb, null_emb)

        for i, block in enumerate(self.up_blocks):
            res = residuals.pop()
            if res is None:
                res = ()
            elif not isinstance(res, tuple):
                res = (res,)
            expected_args = len(getattr(block, 'resnets', []))
            while len(res) < expected_args:
                ref = res[-1] if res else x
                dummy = torch.zeros_like(ref)
                res += (dummy,)
            try:
                x = block(x, res, temb=emb, encoder_hidden_states=null_emb, upsample_size=list(x.shape[-2:]))
            except IndexError:
                print(f"[Warning] UpBlock {i} received {len(res)} residual(s), but expected more. Skipping this block.")
                continue
            x = self.attn_blocks[i](x, cbct_cond)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


# ---- Load Hugging Face Models ---- #
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
base_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(DEVICE)

for param in autoencoder.parameters():
    param.requires_grad = False
for param in base_unet.parameters():
    param.requires_grad = False

model = ControlNetLDM(base_unet, control_channels=4, attn_type='mha').to(DEVICE)
trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
diffusion = Diffusion(device=DEVICE)
optimizer = torch.optim.Adam(trainable_params, lr=1e-4)


# ---- Dataset for Real Images ---- #
class CBCTtoCTDataset(Dataset):
    def __init__(self, CBCT_path, CT_path, image_size, transform=None):
        self.CBCT_slices = self._collect_slices(CBCT_path)
        self.CT_slices = self._collect_slices(CT_path)
        self.image_size = image_size
        self.transform = transform
        
    def _collect_slices(self, dataset_path):
        slice_paths = []
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(subdir_path):
                for slice_name in os.listdir(subdir_path):
                    slice_path = os.path.join(subdir_path, slice_name)
                    slice_paths.append(slice_path)
        return slice_paths
    
    def __len__(self):
        return min(len(self.CBCT_slices), len(self.CT_slices))
    
    def __getitem__(self, idx):
        CBCT_path = self.CBCT_slices[idx]
        CT_path = self.CT_slices[idx]

        CBCT_slice = Image.open(CBCT_path).convert("L")
        CT_slice = Image.open(CT_path).convert("L")
        
        if self.transform:
            CBCT_slice = self.transform(CBCT_slice)
            CT_slice = self.transform(CT_slice)

        return CBCT_slice, CT_slice


# ---- Dataloader Setup ---- #
# probably have to do something similar to 
# x = x.expand(-1, 3, -1, -1) // Expand to 3 channels for compatibility with the autoencoder
def get_dataloaders(config, transform):
    full_dataset = CBCTtoCTDataset(
        CBCT_path=config["paths"]["train_CBCT"],
        CT_path=config["paths"]["train_CT"],
        image_size=config["model"]["image_size"],
        transform=transform
    )

    subset_size = 10  # Adjust for your desired subset size
    subset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])

    val_size = int(len(subset) * 0.2)
    train_size = len(subset) - val_size
    print(f"Splitting {len(subset)} images into {train_size} train images and {val_size} validation images.")

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(subset, [train_size, val_size], generator=generator)

    train_dataloader = DataLoader(train_dataset, 
                            batch_size=config["train"]["batch_size"], 
                            shuffle=True, 
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    val_dataloader = DataLoader(val_dataset,
                            batch_size=config["train"]["batch_size"],
                            shuffle=False,
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    return train_dataloader, val_dataloader


# ---- Training Loop ---- #
def train(dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for cbct, sct in dataloader:
            cbct, sct = cbct.to(DEVICE), sct.to(DEVICE)

            with torch.no_grad():
                cbct_latents = autoencoder.encode(cbct).latent_dist.sample()
                sct_latents = autoencoder.encode(sct).latent_dist.sample()

            t = diffusion.sample_timesteps(cbct.size(0)).float()
            noise = torch.randn_like(sct_latents)
            noisy_latents = diffusion.add_noise(sct_latents, t, noise)

            pred = model(noisy_latents, t, cbct_latents)
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ---- Inference ---- #
def generate_sct(cbct):
    cbct = cbct.to(DEVICE)
    with torch.no_grad():
        cbct_latents = autoencoder.encode(cbct).latent_dist.sample()
        x = torch.randn_like(cbct_latents)
        for t in reversed(range(diffusion.timesteps)):
            x = model(x, torch.tensor([t]*x.size(0), device=DEVICE).float(), cbct_latents)
        sct_pred = autoencoder.decode(x).sample
    return sct_pred


# ---- Dummy Test Driver ---- #
if __name__ == "__main__":
    print("🔧 Starting training with real images...")

    # Set up your configurations and transformations
    config = {
        "paths": {
            "train_CBCT": "path_to_CBCT_images",
            "train_CT": "path_to_CT_images"
        },
        "model": {
            "image_size": 256  # Adjust as needed
        },
        "train": {
            "batch_size": 8,
            "num_workers": 4
        }
    }

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust the image size if needed
        transforms.ToTensor(),  # Convert the images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    train_loader, val_loader = get_dataloaders(config, transform)

    # Start training with real images
    train(train_loader, epochs=10)

    print("✅ Training complete.")
