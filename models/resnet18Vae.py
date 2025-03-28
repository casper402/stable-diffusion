import torch
import torch.nn as nn
from torchvision import models

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
        )

    def forward(self, x):
        return F.silu(x + self.block(x))
    
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, H*W).transpose(1, 2)
        x_attn, _ = self.attention(x_flat, x_flat, x_flat)
        x_attn = x_attn.transpose(1, 2).view(B, C, H, W)
        return x + self.norm(x_attn)

class VAE(nn.Module):
    def __init__(self, latent_dim, in_channels=1, out_channels=1):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        with torch.no_grad():
            avg_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.weight = nn.Parameter(avg_weight.repeat(1, in_channels, 1, 1))
        resnet.conv1 = new_conv1
        resnet.maxpool = nn.Identity()
        resnet_layers = list(resnet.children())

        self.encoder = nn.Sequential(
            *resnet_layers[:8],
            nn.Conv2d(512, 2*latent_dim, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 3, 1, 1),
            ResidualBlock(512),
            AttentionBlock(512),                    # Attention added
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # ->64x64
            ResidualBlock(256),
            AttentionBlock(256),                    # Attention added
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # ->128x128
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),     # ->256x256
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent = self.encoder(x)

        mu, logvar = torch.chunk(latent, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)
        return z, mu, logvar, reconstructed