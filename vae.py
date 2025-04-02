import torch 
import torch.nn as nn
import os
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from piq import ssim
from torchvision.models import vgg16
from torchvision.transforms import Normalize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features[:8].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # adjust if needed

    def forward(self, recon_x, x):
        recon_x = recon_x.repeat(1, 3, 1, 1)  # grayscale -> 3-channel
        x = x.repeat(1, 3, 1, 1)

        recon_x = self.normalize(recon_x)
        x = self.normalize(x)

        feat_recon = self.vgg(recon_x)
        feat_real = self.vgg(x)

        return F.mse_loss(feat_recon, feat_real)

perceptual_loss = PerceptualLoss(device=device)

class SsimLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()

    def normalize(self, x):
        # Normalize fomr [-1, 1] to [0, 1]
        return (x + 1) / 2.0

    def forward(self, recon, x):
        recon = self.normalize(recon)
        x = self.normalize(x)
        return 1 - ssim(recon, x)

ssim_loss = SsimLoss(device=device)

def vae_loss(recon, x, mu, logvar, perceptual_weight=0.1, ssim_weight=0.8, mse_weight=1.0, kl_weight=0.1, l1_weight=0.2):
    mse = F.mse_loss(recon, x)
    perceptual = perceptual_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    ssim_val = ssim_loss(recon, x)
    l1 = F.l1_loss(recon, x)
    total_loss = mse_weight * mse + perceptual_weight * perceptual + kl_weight * kl + ssim_val * ssim_weight + l1 * l1_weight
    return total_loss
    
class CTDataset(Dataset):
    def __init__(self, CT_path):
        self.CT_slices = self._collect_slices(CT_path)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad((0, 64, 0, 64)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
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
        return len(self.CT_slices)
    
    def __getitem__(self, idx):
        CT_path = self.CT_slices[idx]
        CT_slice = Image.open(CT_path).convert("L")
        if self.transform:
            CT_slice = self.transform(CT_slice)
        return CT_slice

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Two 3x3 conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = F.silu(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x)


# -------------------------------------------------
# Downsample Block (factor 2)
# -------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res(x)
        x = self.down(x)  # (B, out_channels, H/2, W/2)
        return x


# -------------------------------------------------
# Upsample Block (factor 2)
# -------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res(x)
        x = self.up(x)  # (B, out_channels, 2H, 2W)
        return x


# -------------------------------------------------
# Encoder (3 downs => factor of 8)
# -------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, latent_channels=4):
        super().__init__()
        # Initial conv: 1 -> 64
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 3 down blocks: (64->128), (128->256), (256->256)
        # You could pick 256->512 for the third block, but 256 is often enough for 8x factor.
        self.down1 = DownBlock(base_channels, base_channels * 2)   # 64 -> 128
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128 -> 256
        self.down3 = DownBlock(base_channels * 4, base_channels * 4)  # 256 -> 256

        # Final 1x1 conv to produce 2*latent_channels => [mu, logvar]
        self.conv_out = nn.Conv2d(base_channels * 4, latent_channels * 2, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)    # (B,64,256,256)
        x = self.down1(x)      # (B,128,128,128)
        x = self.down2(x)      # (B,256,64,64)
        x = self.down3(x)      # (B,256,32,32)

        out = self.conv_out(x) # (B,8,32,32)
        mu, logvar = torch.chunk(out, 2, dim=1)  # (B,4,32,32) each
        return mu, logvar


# -------------------------------------------------
# Decoder (3 ups => factor of 8)
# -------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=64, latent_channels=4):
        super().__init__()
        # Project from 4 => (base_channels*4=256) at 32x32
        self.conv_in = nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1)

        # 3 up blocks: mirror the 3 downs
        self.up1 = UpBlock(base_channels * 4, base_channels * 4)    # 256 -> 256
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)    # 256 -> 128
        self.up3 = UpBlock(base_channels * 2, base_channels)        # 128 -> 64

        # Final conv to get single grayscale channel
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)   # (B,256,32,32)
        x = self.up1(x)       # (B,256,64,64)
        x = self.up2(x)       # (B,128,128,128)
        x = self.up3(x)       # (B,64,256,256)
        x = self.conv_out(x)  # (B,1,256,256)
        return torch.tanh(x)


# -------------------------------------------------
# Full AutoencoderKL
# -------------------------------------------------
class AutoencoderKL(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, latent_channels=4):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, latent_channels)
        self.decoder = Decoder(out_channels, base_channels, latent_channels)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return z, mu, logvar, recon

dataset = CTDataset('../training_data/CT')
subset, _ = random_split(dataset, [500, len(dataset) - 500])

train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size - 10
test_size = 10
train_dataset, val_dataset, test_dataset = random_split(subset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

vae = AutoencoderKL().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

best_val_loss = float('inf')
save_path = 'best_vae_ct_2.pth'
for epoch in range(1000):
    vae.train()
    train_loss = 0

    for x in train_loader:
        x = x.to(device)
        _, mu, logvar, recon = vae(x)
        loss = vae_loss(recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            _, mu, logvar, recon = vae(x)
            loss = vae_loss(recon, x, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        conuter = 0
        torch.save(vae.state_dict(), save_path)
        print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

    if (epoch+0) % 50 == 0:
        vae.eval()
        pred_dir = f"./predictions_2/epoch_{epoch+1}/"
        os.makedirs(pred_dir, exist_ok=True)

        print("Saving predictions")

        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(device)
                _, _, _, recon = vae(x)

                for j in range(min(x.size(0), 8)):
                    original = x[j]
                    reconstructed = recon[j]

                    vutils.save_image(original, f"{pred_dir}/img_{i}_{j}_orig.png", normalize=True, value_range=(-1, 1))
                    vutils.save_image(reconstructed, f"{pred_dir}/img_{i}_{j}_recon.png", normalize=True, value_range=(-1, 1))

                if i >= 2:  # Save only a few batches
                    break

vae.load_state_dict(torch.load(save_path))
vae.eval()

# Inference loop on test_loader
pred_dir = "./predictions_2/"
os.makedirs(pred_dir, exist_ok=True)

with torch.no_grad():
    for i, x in enumerate(test_loader):
        x = x.to(device)
        _ , _, _, recon = vae(x)

        # Save the first few predictions
        for j in range(min(x.size(0), 8)):
            original = x[j]
            reconstructed = recon[j]

            # Save originals and reconstructions as image files
            vutils.save_image(original, f"{pred_dir}/img_{i}_{j}_orig.png", normalize=True)
            vutils.save_image(reconstructed, f"{pred_dir}/img_{i}_{j}_recon.png", normalize=True)

        # Optional: stop after a few batches
        if i >= 2:
            break