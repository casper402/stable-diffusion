import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000.0) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.act(self.linear1(emb))
        emb = self.linear2(emb)
        return emb
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_groups=32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5 # 1 / sqrt(dim_head)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        res = x 
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.reshape(b, self.num_heads, c // self.num_heads, h * w), qkv
        )
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return self.to_out(out) + res


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, time_emb_dim=None, dropout_rate=0.1, num_groups=32):
        super().__init__()
        out_channels = out_channels or in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.dropout = nn.Dropout(dropout_rate)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_emb_proj is not None and t_emb is not None:
            h += self.time_emb_proj(t_emb)[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        out = F.silu(h + self.residual(x))
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attn = AttentionBlock(out_channels, num_heads) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(skip)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, has_attn=False, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(out_channels + skip_channels, out_channels, time_emb_dim, dropout_rate)
        self.attn = AttentionBlock(out_channels, num_heads) if has_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 base_channels=64, 
                 time_emb_dim=256, 
                 num_heads=8,
                 dropout_rate=0.1):
        super().__init__()
        
        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        ch1 = base_channels       # 64
        ch2 = base_channels * 2   # 128
        ch3 = base_channels * 4   # 256
        ch4 = base_channels * 8   # 512

        attn_level_0 = False # Corresponds to ch2 (128 channels)
        attn_level_1 = True  # Corresponds to ch3 (256 channels)
        attn_level_2 = True  # Corresponds to ch4 (512 channels)

        self.down1 = DownBlock(ch1, ch2, time_emb_dim, has_attn=attn_level_0, num_heads=num_heads, dropout_rate=dropout_rate)
        self.down2 = DownBlock(ch2, ch3, time_emb_dim, has_attn=attn_level_1, num_heads=num_heads, dropout_rate=dropout_rate)
        self.down3 = DownBlock(ch3, ch4, time_emb_dim, has_attn=attn_level_2, num_heads=num_heads, dropout_rate=dropout_rate)

        self.bottleneck_res1 = ResBlock(ch4, ch4, time_emb_dim, dropout_rate=dropout_rate)
        self.bottleneck_attn = AttentionBlock(ch4, num_heads=num_heads) if attn_level_2 else nn.Identity()
        self.bottleneck_res2 = ResBlock(ch4, ch4, time_emb_dim, dropout_rate=dropout_rate)

        self.up3 = UpBlock(ch4, ch4, ch3, time_emb_dim, has_attn=attn_level_2, num_heads=num_heads, dropout_rate=dropout_rate)
        self.up2 = UpBlock(ch3, ch3, ch2, time_emb_dim, has_attn=attn_level_1, num_heads=num_heads, dropout_rate=dropout_rate)
        self.up1 = UpBlock(ch2, ch2, ch1, time_emb_dim, has_attn=attn_level_0, num_heads=num_heads, dropout_rate=dropout_rate)

        self.final_norm = nn.GroupNorm(8, ch1) # Norm before final conv
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.init_conv(x)         # Initial convolution: [B, 4, H, W] -> [B, 64, H, W]

        x1, skip1 = self.down1(x, t_emb)   # -> [B, 128, H/2, W/2]
        x2, skip2 = self.down2(x1, t_emb)  # -> [B, 256, H/4, W/4]
        x3, skip3 = self.down3(x2, t_emb)  # -> [B, 512, H/8, W/8]

        xb = self.bottleneck_res1(x3, t_emb)
        xb = self.bottleneck_attn(xb)
        xb = self.bottleneck_res2(xb, t_emb) # -> [B, 512, H/8, W/8]

        x = self.up3(xb, skip3, t_emb)    # -> [B, 256, H/4, W/4]
        x = self.up2(x, skip2, t_emb)     # -> [B, 128, H/2, W/2]
        x = self.up1(x, skip1, t_emb)     # -> [B, 64, H, W]

        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)            # -> [B, 4, H, W]
        return x
