import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from diffusers import AutoencoderKL, UNet2DModel
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Diffusion Schedule ---- #
class Diffusion:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(DEVICE)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_cumprod_t = self.alpha_cumprod[t.long()].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x0 + torch.sqrt(1 - alpha_cumprod_t) * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=DEVICE)

# ---- Dataset ---- #
class PairedImageDataset(Dataset):
    def __init__(self, cbct_dir, sct_dir, transform=None):
        self.cbct_paths = sorted([os.path.join(cbct_dir, f) for f in os.listdir(cbct_dir)])
        self.sct_paths = sorted([os.path.join(sct_dir, f) for f in os.listdir(sct_dir)])
        self.transform = transform

    def __len__(self):
        return min(len(self.cbct_paths), len(self.sct_paths))

    def __getitem__(self, idx):
        cbct = Image.open(self.cbct_paths[idx]).convert("L")
        sct = Image.open(self.sct_paths[idx]).convert("L")
        if self.transform:
            cbct = self.transform(cbct)
            sct = self.transform(sct)
        return cbct.expand(3, -1, -1), sct.expand(3, -1, -1)

# ---- Setup Models ---- #
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
unet = UNet2DModel(
    sample_size=32,  # Adjust based on your latent resolution
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
).to(DEVICE)

for p in autoencoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
diffusion = Diffusion(timesteps=1000)

# ---- Transform ---- #
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# ---- Training Loop ---- #
def train(dataloader, epochs=5):
    unet.train()
    for epoch in range(epochs):
        for step, (cbct, sct) in enumerate(dataloader):
            sct = sct.to(DEVICE)

            with torch.no_grad():
                latents = autoencoder.encode(sct).latent_dist.sample() * 0.18215

            t = diffusion.sample_timesteps(sct.size(0)).float()
            noise = torch.randn_like(latents)
            noisy_latents = diffusion.add_noise(latents, t, noise)

            pred = unet(noisy_latents, timestep=t)["sample"]
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

# ---- Inference ---- #
def sample(latent_shape=(1, 4, 32, 32)):
    unet.eval()
    x = torch.randn(latent_shape).to(DEVICE)
    with torch.no_grad():
        for t in reversed(range(diffusion.timesteps)):
            x = unet(x, timestep=torch.tensor([t], device=DEVICE).float())["sample"]
        img = autoencoder.decode(x / 0.18215).sample
    return img

# ---- Run It ---- #
if __name__ == "__main__":
    config = {
        "cbct_dir": "path/to/cbct",
        "sct_dir": "path/to/sct",
        "batch_size": 2
    }

    dataset = PairedImageDataset(config["cbct_dir"], config["sct_dir"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    train(dataloader, epochs=1)

    result = sample()
    transforms.ToPILImage()(result.squeeze().cpu().clamp(-1, 1) * 0.5 + 0.5).show()
