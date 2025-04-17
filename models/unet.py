import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.act(self.linear1(emb))
        emb = self.linear2(emb)
        return emb

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, time_emb_dim=None, dropout_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.residual(x)
        return x + h
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = Normalize(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True # Crucial for our reshaping (B, L, E)
        )
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        b, c, h, w = h_.shape
        h_ = h_.view(b, c, h * w).transpose(1, 2) # Now shape (B, H*W, C)
        h_, _ = self.mha(h_, h_, h_) # Output shape (B, H*W, C)
        h_ = h_.transpose(1, 2).view(b, c, h, w)
        return x + h_
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.heads = num_heads
        self.scale = dim ** -0.5
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        q = self.to_q(x).reshape(b, self.heads, c // self.heads, h * w)
        k = self.to_k(context).reshape(b, self.heads, c // self.heads, -1)
        v = self.to_v(context).reshape(b, self.heads, c // self.heads, -1)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, c, h, w)
        return self.to_out(out + x)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, has_attn=False, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.has_attn = has_attn

        self.res_block1 = ResnetBlock(in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention1 = AttentionBlock(out_channels, num_heads=num_heads) if has_attn else None
        self.res_block2 = ResnetBlock(out_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention2 = AttentionBlock(out_channels, num_heads=num_heads) if has_attn else None
        self.downsample = Downsample(out_channels, with_conv=True)

    def forward(self, x, temb=None):
        h = x
        h = self.res_block1(h, temb)
        if self.has_attn:
            h = self.attention1(h)
        h = self.res_block2(h, temb)
        if self.has_attn:
            h = self.attention2(h)
        skip = h
        h = self.downsample(h)
        return h, skip
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_in, time_emb_dim=None, has_attn=False, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.has_attn = has_attn

        self.upsample = Upsample(in_channels, with_conv=True)
        self.res_block1 = ResnetBlock(in_channels + skip_in, out_channels, time_emb_dim, dropout_rate)
        self.attention1 = AttentionBlock(out_channels, num_heads=num_heads) if has_attn else None
        self.res_block2 = ResnetBlock(out_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention2 = AttentionBlock(out_channels, num_heads=num_heads) if has_attn else None

    def forward(self, x, skip, temb=None):
        x = self.upsample(x)
        h = torch.cat([x, skip], dim=1)
        h = self.res_block1(h, temb)
        h = self.attention1(h)
        h = self.res_block2(h, temb)
        h = self.attention2(h)
        return h
    
class MiddleBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        self.res_block1 = ResnetBlock(in_channels, in_channels, time_emb_dim, dropout)
        self.attention1 = CrossAttentionBlock(in_channels, num_heads=num_heads)
        self.res_block2 = ResnetBlock(in_channels, in_channels, time_emb_dim, dropout)

    def forward(self, x, context, temb=None):
        h = x
        h = self.res_block1(h, temb)
        h = self.attention1(h, context)
        h = self.res_block2(h, temb)
        return h

class UNet(nn.Module): # Modified UNet for ControlNet
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 base_channels=256, 
                 time_emb_dim=1024, 
                 num_heads=16,
                 dropout_rate=0.1):
        super().__init__()
        
        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        ch1 = base_channels * 1
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4

        attn_level_0 = True # Corresponds to ch2
        attn_level_1 = True  # Corresponds to ch3
        attn_level_2 = True  # Corresponds to ch4

        self.down1 = DownBlock(ch1, ch2, time_emb_dim, attn_level_0, num_heads, dropout_rate)
        self.down2 = DownBlock(ch2, ch3, time_emb_dim, attn_level_1, num_heads, dropout_rate)
        self.down3 = DownBlock(ch3, ch4, time_emb_dim, attn_level_2, num_heads, dropout_rate)

        self.context_proj = nn.Conv2d(in_channels, ch4, kernel_size=1)
        self.middle = MiddleBlock(ch4, time_emb_dim, num_heads, dropout_rate)

        self.up3 = UpBlock(ch4, ch3, ch4, time_emb_dim, attn_level_2, num_heads, dropout_rate)
        self.up2 = UpBlock(ch3, ch2, ch3, time_emb_dim, attn_level_1, num_heads, dropout_rate)
        self.up1 = UpBlock(ch2, ch1, ch2, time_emb_dim, attn_level_0, num_heads, dropout_rate)

        self.final_norm = Normalize(ch1) # Norm before final conv
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, context=None, t=None):
        t_emb = self.time_embedding(t)
        x = self.init_conv(x)         # Initial convolution: [B, 4, H, W] -> [B, 64, H, W]
        x, skip1 = self.down1(x, t_emb)   # -> [B, 128, H/2, W/2]
        x, skip2 = self.down2(x, t_emb)  # -> [B, 256, H/4, W/4]
        x, skip3 = self.down3(x, t_emb)  # -> [B, 512, H/8, W/8]

        if context is None:
            context = x
        else:
            context = self.context_proj(context)
            
        x = self.middle(x, context, t_emb) # -> [B, 512, H/8, W/8]

        x = self.up3(x, skip3, t_emb)    # -> [B, 256, H/4, W/4]
        x = self.up2(x, skip2, t_emb)     # -> [B, 128, H/2, W/2]
        x = self.up1(x, skip1, t_emb)

        x = self.final_norm(x)
        x = nonlinearity(x)
        x = self.final_conv(x) # -> [B, 4, H, W]
        return x
