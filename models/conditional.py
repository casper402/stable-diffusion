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
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True # Crucial for our reshaping (B, L, E)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        res = x
        x = self.norm(x)
        x = x.view(b, c, h * w).transpose(1, 2) # Now shape (B, H*W, C)
        attn_output, _ = self.mha(x, x, x) # Output shape (B, H*W, C)
        attn_output = attn_output.transpose(1, 2).view(b, c, h, w)
        return attn_output + res

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
    def __init__(self, in_channels, out_channels, time_emb_dim=None, has_attn=False, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attn = AttentionBlock(out_channels, num_heads) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        x = self.res(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(skip)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim=None, has_attn=False, num_heads=8, dropout_rate=0.1):
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
    
class SimplePACALayer(nn.Module):
    def __init__(self, channels, num_heads=8, num_groups=32):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        if channels % num_groups != 0:
            num_groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else (2 if channels % 2 == 0 else 1))
            if channels % num_groups != 0: num_groups = 1

        self.num_heads = num_heads

        self.norm_q = nn.GroupNorm(num_groups, channels)
        self.norm_kv = nn.GroupNorm(num_groups, channels)

        self.to_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, q_features, kv_features):
        res = q_features
        b, c, h, w = q_features.shape
        q = self.norm_q(q_features)
        kv = self.norm_kv(kv_features)
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        attn_output, _ = self.mha(query=q, key=k, value=v)
        attn_output = attn_output.transpose(1, 2).view(b, c, h, w)
        return self.to_out(attn_output) + res
        
class ControlNet(nn.Module):
    def __init__(self,
                 in_channels=1, # Input channels for condition (e.g., CBCT)
                 base_channels=128,
                 num_heads=16,
                 dropout_rate=0.1,
                 unet_channels=(256, 512, 1024)): # ch1, ch2, ch3, ch4 from UNet
        super().__init__()

        unet_ch1, unet_ch2, unet_ch3 = unet_channels

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        ctrl_ch1 = base_channels
        ctrl_ch2 = base_channels * 2
        ctrl_ch3 = base_channels * 4
        ctrl_ch4 = base_channels * 8

        attn_level_0 = False
        attn_level_1 = True
        attn_level_2 = True

        self.down1 = DownBlock(ctrl_ch1, ctrl_ch2, time_emb_dim=None, has_attn=attn_level_0, num_heads=num_heads, dropout_rate=dropout_rate)
        self.down2 = DownBlock(ctrl_ch2, ctrl_ch3, time_emb_dim=None, has_attn=attn_level_1, num_heads=num_heads, dropout_rate=dropout_rate)
        self.down3 = DownBlock(ctrl_ch3, ctrl_ch4, time_emb_dim=None, has_attn=attn_level_2, num_heads=num_heads, dropout_rate=dropout_rate)

        # self.bottleneck_res1 = ResBlock(ctrl_ch4, ctrl_ch4, time_emb_dim=None, dropout_rate=dropout_rate)
        # self.bottleneck_attn = AttentionBlock(ctrl_ch4, num_heads=num_heads) if attn_level_2 else nn.Identity()
        # self.bottleneck_res2 = ResBlock(ctrl_ch4, ctrl_ch4, time_emb_dim=None, dropout_rate=dropout_rate)

        # Projection layers to match UNet decoder output dimensions for SimplePACALayer
        self.proj_c1 = nn.Conv2d(ctrl_ch2, unet_ch1, kernel_size=1) # Project down1 output (ctrl_ch2) to unet_ch1
        self.proj_c2 = nn.Conv2d(ctrl_ch3, unet_ch2, kernel_size=1) # Project down2 output (ctrl_ch3) to unet_ch2
        self.proj_c3 = nn.Conv2d(ctrl_ch4, unet_ch3, kernel_size=1) # Project down3 output (ctrl_ch4) to unet_ch3
        # Bottleneck projection might be needed if bottleneck PACA is used
        # self.proj_cb = nn.Conv2d(ctrl_ch4, unet_ch4, kernel_size=1)

    def forward(self, x_condition):
        x = self.init_conv(x_condition)

        x1_ctrl_orig, _ = self.down1(x)
        x2_ctrl_orig, _ = self.down2(x1_ctrl_orig)
        x3_ctrl_orig, _ = self.down3(x2_ctrl_orig)

        # xb_ctrl = self.bottleneck_res1(x3_ctrl_orig)
        # xb_ctrl = self.bottleneck_attn(xb_ctrl)
        # xb_ctrl = self.bottleneck_res2(xb_ctrl)

        # Project features to match UNet decoder dims for SimplePACALayer
        c1 = self.proj_c1(x1_ctrl_orig)
        c2 = self.proj_c2(x2_ctrl_orig)
        c3 = self.proj_c3(x3_ctrl_orig)
        # cb = xb_ctrl # Bottleneck features often not projected unless used by PACA

        return c1, c2, c3

class UNet(nn.Module): # Modified UNet for ControlNet
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 base_channels=128, 
                 time_emb_dim=256, 
                 num_heads=16,
                 dropout_rate=0.1):
        super().__init__()
        
        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 8

        attn_level_0 = False # Corresponds to ch2
        attn_level_1 = True  # Corresponds to ch3
        attn_level_2 = True  # Corresponds to ch4

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

        self.paca1 = SimplePACALayer(channels=ch1, num_heads=num_heads)
        self.paca2 = SimplePACALayer(channels=ch2, num_heads=num_heads)
        self.paca3 = SimplePACALayer(channels=ch3, num_heads=num_heads)


    def forward(self, x, t, control_features):
        c1, c2, c3 = control_features

        t_emb = self.time_embedding(t)
        x = self.init_conv(x)         # Initial convolution: [B, 4, H, W] -> [B, 64, H, W]

        x, skip1 = self.down1(x, t_emb)   # -> [B, 128, H/2, W/2]
        x, skip2 = self.down2(x, t_emb)  # -> [B, 256, H/4, W/4]
        x, skip3 = self.down3(x, t_emb)  # -> [B, 512, H/8, W/8]

        x = self.bottleneck_res1(x, t_emb)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_res2(x, t_emb) # -> [B, 512, H/8, W/8]

        x = self.up3(x, skip3, t_emb)    # -> [B, 256, H/4, W/4]
        x = self.paca3(q_features=x, kv_features=c3)
        x = self.up2(x, skip2, t_emb)     # -> [B, 128, H/2, W/2]
        x = self.paca2(q_features=x, kv_features=c2)
        x = self.up1(x, skip1, t_emb)
        x = self.paca1(q_features=x, kv_features=c1)     # -> [B, 64, H, W]

        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)            # -> [B, 4, H, W]
        return x
