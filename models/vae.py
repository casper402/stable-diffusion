import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Encoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),  # -> 128x128
            ResidualBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1), # -> 64x64
            ResidualBlock(256),
            AttentionBlock(256),
            nn.Conv2d(256, 512, 4, 2, 1),  # -> 32x32
            ResidualBlock(512),
            AttentionBlock(512),
        )
        self.conv_mu = nn.Conv2d(512, latent_dim, 1)
        self.conv_logvar = nn.Conv2d(512, latent_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
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
        )

    def forward(self, z):
        x = self.decoder(z)
        return torch.sigmoid(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return z, mu, logvar, recon
