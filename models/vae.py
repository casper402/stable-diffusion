import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res(x)
        x = self.down(x)  # (B, out_channels, H/2, W/2)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res(x)
        x = self.up(x)  # (B, out_channels, 2H, 2W)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, latent_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels * 2)   
        self.down2 = DownBlock(base_channels * 2, base_channels * 4) 
        self.down3 = DownBlock(base_channels * 4, base_channels * 8) 

        # Final 1x1 conv to produce 2*latent_channels => [mu, logvar]
        self.conv_out = nn.Conv2d(base_channels * 8, latent_channels * 2, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x) 
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        out = self.conv_out(x)
        mu, logvar = torch.chunk(out, 2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=32, latent_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, base_channels * 8, kernel_size=3, padding=1)

        # 3 up blocks: mirror the 3 downs
        self.up1 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        # Final conv to get single grayscale channel
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(x)
        return torch.tanh(x)

class VAE(nn.Module):
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