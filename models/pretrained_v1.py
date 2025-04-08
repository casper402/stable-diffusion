# This file is a simplified model that just uses a pretrained network, but not much else

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel

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

        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)


# ---- ControlNet-style Conditioning ---- #
class ControlBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class ControlNetLDM(nn.Module):
    def __init__(self, base_unet, control_branch):
        super().__init__()
        self.unet = base_unet
        self.control_branch = control_branch

    def forward(self, x, t, cbct_cond):
        control_hint = self.control_branch(cbct_cond)
        x = x + control_hint  # Use additive conditioning
        return self.unet(x, t, encoder_hidden_states=None).sample


# ---- Load Hugging Face Models ---- #
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
base_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(DEVICE)

# Wrap in our conditional ControlNet model
control_branch = ControlBranch(in_channels=1, out_channels=4).to(DEVICE)
model = ControlNetLDM(base_unet, control_branch).to(DEVICE)

diffusion = Diffusion(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ---- Training Loop ---- #
def train(dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for cbct, sct in dataloader:
            cbct, sct = cbct.to(DEVICE), sct.to(DEVICE)

            with torch.no_grad():
                cbct_latents = autoencoder.encode(cbct).latent_dist.sample()
                sct_latents = autoencoder.encode(sct).latent_dist.sample()

            t = diffusion.sample_timesteps(cbct.size(0))
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
            x = model(x, torch.tensor([t]*x.size(0), device=DEVICE), cbct_latents)
        sct_pred = autoencoder.decode(x).sample
    return sct_pred
