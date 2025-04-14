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

    def forward(self, timesteps): #TODO: Use time embedding implementation form ldm: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L218
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
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, has_attn=False, num_heads=16, dropout_rate=0.1):
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
    def __init__(self, in_channels, out_channels, skip_in, time_emb_dim=None, has_attn=False, num_heads=16, dropout_rate=0.1):
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
    def __init__(self, in_channels, time_emb_dim=None, num_heads=16, dropout=0.1):
        super().__init__()
        self.res_block1 = ResnetBlock(in_channels, in_channels, time_emb_dim, dropout)
        self.attention1 = AttentionBlock(in_channels, num_heads=num_heads)
        self.res_block2 = ResnetBlock(in_channels, in_channels, time_emb_dim, dropout)

    def forward(self, x, temb=None):
        h = x
        h = self.res_block1(h, temb)
        h = self.attention1(h)
        h = self.res_block2(h, temb)
        return h
    
class PACALayer(nn.Module):
    def __init__(self, query_dim, kv_dim, num_heads=8, num_groups=32):
        super().__init__()
        if query_dim <= 0 or kv_dim <= 0:
             raise ValueError("Query and KV dimensions must be positive")
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")
        # Ensure num_groups is valid for both dims, potentially choosing the smaller constraint
        q_num_groups = num_groups
        if query_dim % q_num_groups != 0:
            q_num_groups = 8 if query_dim % 8 == 0 else (4 if query_dim % 4 == 0 else (2 if query_dim % 2 == 0 else 1))
            if query_dim % q_num_groups != 0: q_num_groups = 1
        kv_num_groups = num_groups
        if kv_dim % kv_num_groups != 0:
            kv_num_groups = 8 if kv_dim % 8 == 0 else (4 if kv_dim % 4 == 0 else (2 if kv_dim % 2 == 0 else 1))
            if kv_dim % kv_num_groups != 0: kv_num_groups = 1

        self.num_heads = num_heads

        # Norm for query (UNet features) and key/value (ControlNet features)
        self.norm_q = nn.GroupNorm(q_num_groups, query_dim)
        self.norm_kv = nn.GroupNorm(kv_num_groups, kv_dim)

        # Projections: Q from query_dim, K and V from kv_dim
        self.to_q = nn.Conv2d(query_dim, query_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(kv_dim, query_dim, kernel_size=1, bias=False) # Project K to query_dim
        self.to_v = nn.Conv2d(kv_dim, query_dim, kernel_size=1, bias=False) # Project V to query_dim

        # Use MultiheadAttention with separate K, V inputs (projected to query_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=query_dim, # Attention operates in query dimension space
            num_heads=num_heads,
            batch_first=True
            # kdim and vdim args are not needed as we project K, V manually first
        )

        # Output projection
        self.to_out = nn.Conv2d(query_dim, query_dim, kernel_size=1)

    def forward(self, q_features, kv_features):
        # q_features: from UNet decoder (Query)
        # kv_features: from ControlNet encoder (Key, Value)
        res = q_features # Residual connection starts from query features
        b, c_q, h, w = q_features.shape
        _, c_kv, _, _ = kv_features.shape # H, W assumed to match q_features

        # Normalize features
        q = self.norm_q(q_features)
        kv = self.norm_kv(kv_features)

        # Project to Q, K, V (K, V projected to query dimension c_q)
        q = self.to_q(q)
        k = self.to_k(kv)
        v = self.to_v(kv)

        # Reshape for MultiheadAttention: (B, C, H, W) -> (B, L, C) where L=H*W
        q = q.view(b, c_q, h * w).transpose(1, 2)
        k = k.view(b, c_q, h * w).transpose(1, 2) # K is now shape (B, L, C_q)
        v = v.view(b, c_q, h * w).transpose(1, 2) # V is now shape (B, L, C_q)

        # Apply cross-attention
        attn_output, _ = self.mha(query=q, key=k, value=v) # Output shape (B, L, C_q)

        # Reshape back: (B, L, C) -> (B, C, L) -> (B, C, H, W)
        attn_output = attn_output.transpose(1, 2).view(b, c_q, h, w)

        # Final projection and residual connection
        return self.to_out(attn_output) + res

class ControlNet(nn.Module):
    def __init__(self,
                 in_channels=4, # Input channels for condition (e.g., CBCT)
                 base_channels=256,
                 num_heads=16,
                 dropout_rate=0.1,
                 unet_channels=(256, 512, 1024)): # ch1, ch2, ch3 from UNet
        super().__init__()

        unet_ch1, unet_ch2, unet_ch3 = unet_channels

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        ctrl_ch1 = base_channels * 1
        ctrl_ch2 = base_channels * 2
        ctrl_ch3 = base_channels * 4
        ctrl_ch4 = base_channels * 4

        attn_level_0 = True
        attn_level_1 = True
        attn_level_2 = True

        self.down1 = DownBlock(ctrl_ch1, ctrl_ch2, time_emb_dim=None, has_attn=attn_level_0, num_heads=num_heads, dropout_rate=dropout_rate) #32x32x128 -> 16x16x256
        self.down2 = DownBlock(ctrl_ch2, ctrl_ch3, time_emb_dim=None, has_attn=attn_level_1, num_heads=num_heads, dropout_rate=dropout_rate)
        self.down3 = DownBlock(ctrl_ch3, ctrl_ch4, time_emb_dim=None, has_attn=attn_level_2, num_heads=num_heads, dropout_rate=dropout_rate)

        # Projection layers to match UNet decoder output dimensions for SimplePACALayer
        self.proj_c1 = nn.Conv2d(ctrl_ch2, unet_ch1, kernel_size=1) # Project down1 output (ctrl_ch2) to unet_ch1
        self.proj_c2 = nn.Conv2d(ctrl_ch3, unet_ch2, kernel_size=1) # Project down2 output (ctrl_ch3) to unet_ch2
        self.proj_c3 = nn.Conv2d(ctrl_ch4, unet_ch3, kernel_size=1) # Project down3 output (ctrl_ch4) to unet_ch3
      
    def forward(self, x_condition):
        x = self.init_conv(x_condition)

        x1, _ = self.down1(x)
        x2, _ = self.down2(x1)
        x3, _ = self.down3(x2)

        # Project features to match UNet decoder dims for SimplePACALayer
        c1 = self.proj_c1(x1)
        c2 = self.proj_c2(x2)
        c3 = self.proj_c3(x3)
        return c1, c2, c3

class UNetPACA(nn.Module): # Modified UNet for ControlNet
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4, 
                 base_channels=256, 
                 time_emb_dim=1024, # Commonly set to 4*base_channels
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

        self.middle = MiddleBlock(ch4, time_emb_dim, num_heads, dropout_rate)

        self.up3 = UpBlock(ch4, ch3, ch4, time_emb_dim, attn_level_2, num_heads, dropout_rate)
        self.up2 = UpBlock(ch3, ch2, ch3, time_emb_dim, attn_level_1, num_heads, dropout_rate)
        self.up1 = UpBlock(ch2, ch1, ch2, time_emb_dim, attn_level_0, num_heads, dropout_rate)

        self.paca1 = PACALayer(query_dim=ch1, kv_dim=ch1, num_heads=num_heads)
        self.paca2 = PACALayer(query_dim=ch2, kv_dim=ch2, num_heads=num_heads)
        self.paca3 = PACALayer(query_dim=ch3, kv_dim=ch3, num_heads=num_heads)

        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, control_features):
        c1, c2, c3 = control_features

        t_emb = self.time_embedding(t)
        x = self.init_conv(x)         # Initial convolution: [B, 4, H, W] -> [B, 64, H, W]

        x, skip1 = self.down1(x, t_emb)   # -> [B, 128, H/2, W/2]
        x, skip2 = self.down2(x, t_emb)  # -> [B, 256, H/4, W/4]
        x, skip3 = self.down3(x, t_emb)  # -> [B, 512, H/8, W/8]

        x = self.middle(x, t_emb)

        x = self.up3(x, skip3, t_emb)    # -> [B, 256, H/4, W/4]

        q_features_paca3 = x
        kv_features_paca3 = F.interpolate(c3, size=q_features_paca3.shape[-2:], mode='nearest')
        x = self.paca3(q_features=q_features_paca3, kv_features=kv_features_paca3)

        x = self.up2(x, skip2, t_emb)     # -> [B, 128, H/2, W/2]

        q_features_paca2 = x
        kv_features_paca2 = F.interpolate(c2, size=q_features_paca2.shape[-2:], mode='nearest')
        x = self.paca2(q_features=q_features_paca2, kv_features=kv_features_paca2)

        x = self.up1(x, skip1, t_emb)

        q_features_paca1 = x
        kv_features_paca1 = F.interpolate(c1, size=q_features_paca1.shape[-2:], mode='nearest')
        x = self.paca1(q_features=q_features_paca1, kv_features=kv_features_paca1)     # -> [B, 64, H, W]

        x = self.final_norm(x)
        x = nonlinearity(x)
        x = self.final_conv(x) # -> [B, 4, H, W]
        return x
    
class DegradationRemovalModuleResnet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, final_out_channels=4, dropout_rate=0.1):
        super().__init__()

        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 8

        self.init_conv = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)

        self.down1 = DownBlock(ch1, ch2, time_emb_dim=None, has_attn=False, num_heads=None, dropout_rate=dropout_rate) # 256x256 -> 128x128
        self.to_grayscale_128 = nn.Conv2d(ch2, 1, kernel_size=1) # Prediction head for 128x128

        self.down2 = DownBlock(ch2, ch3, time_emb_dim=None, has_attn=False, num_heads=None, dropout_rate=dropout_rate) # 128x128 -> 64x64
        self.to_grayscale_64 = nn.Conv2d(ch3, 1, kernel_size=1) # Prediction head for 64x64

        self.down3 = DownBlock(ch3, ch4, time_emb_dim=None, has_attn=False, num_heads=None, dropout_rate=dropout_rate) # 64x64 -> 32x32
        self.to_grayscale_32 = nn.Conv2d(ch4, 1, kernel_size=1) # Prediction head for 32x32

        self.final_norm = Normalize(ch4)
        self.final_conv = nn.Conv2d(ch1, final_out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.init_conv(x)

        # Block 1 (-> 128x128)
        x = self.down1(x)
        pred_128 = self.to_grayscale_128(x) # Prediction for loss

        # Block 2 (-> 64x64)
        x = self.down2(x) # -> [B, ch3, 64, 64]
        pred_64 = self.to_grayscale_64(x) # Prediction for loss

        # Block 3 (-> 32x32)
        x = self.down3(x) # -> [B, ch4, 32, 32]
        pred_32 = self.to_grayscale_32(x) # Prediction for loss

        # Final output projection for ControlNet
        x = self.final_norm(x)
        x = nonlinearity(x)
        x = self.final_conv(x) # -> [B, 4, 32, 32]

        intermediate_preds = (pred_128, pred_64, pred_32)

        return x, intermediate_preds
