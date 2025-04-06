import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.act(self.linear1(emb))
        emb = self.linear2(emb)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, time_emb_dim=None):
        super().__init__()
        out_channels = out_channels or in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_emb_proj is not None and t_emb is not None:
            h += self.time_emb_proj(t_emb)[:, :, None, None]
        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        out = F.silu(h + self.residual(x))
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_emb_dim)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        skip = self.res(x, t_emb)  # Save skip connection BEFORE downsampling
        x = self.down(skip)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(out_channels + skip_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)

        self.bottleneck = ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        self.up3 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, time_emb_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.init_conv(x)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        x = self.bottleneck(x, t_emb)

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        return self.final_conv(x)
