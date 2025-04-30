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

class TimestepEmbedding(nn.Module): # TODO: Look into this implementation. Compare to ldm: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L218
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, timesteps): #TODO: Use time embedding implementation form 
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
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.residual(x)
        return x + h
    
class AttentionBlock(nn.Module): # From LDM paper
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)
        return x+h_
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, has_attn=False, dropout_rate=0.1, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.has_attn = has_attn

        self.res_block1 = ResnetBlock(in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention1 = AttentionBlock(out_channels) if has_attn else None
        self.res_block2 = ResnetBlock(out_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention2 = AttentionBlock(out_channels) if has_attn else None
        self.downsample = Downsample(out_channels, with_conv=True) if downsample else None

    def forward(self, x, temb=None): # TODO: Return 2 or 3 residuals? If 3 then how to consume the third?
        res_samples = ()
        h = x
        h = self.res_block1(h, temb)
        if self.has_attn:
            h = self.attention1(h)
        res_samples += (h,)
        h = self.res_block2(h, temb)
        if self.has_attn:
            h = self.attention2(h)
        res_samples += (h,)
        if self.downsample:
            h = self.downsample(h)
        return h, res_samples
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None, time_emb_dim=None, has_attn=False, dropout_rate=0.1, upsample=True):
        super().__init__()
        self.has_attn = has_attn
        self.upsample = upsample

        res1_in_channels = in_channels + skip_channels if skip_channels is not None else in_channels
        res2_in_channels = out_channels + skip_channels if skip_channels is not None else out_channels

        self.res_block1 = ResnetBlock(res1_in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention1 = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.res_block2 = ResnetBlock(res2_in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention2 = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.upsample = Upsample(out_channels, with_conv=True) if upsample else None

    def forward(self, x, skips=None, temb=None):
        if skips is not None:
            x = torch.cat([x, skips[0]], dim=1)
        h = self.res_block1(x, temb)
        h = self.attention1(h)
        if skips is not None:
            h = torch.cat([h, skips[1]], dim=1)
        h = self.res_block2(h, temb)
        h = self.attention2(h)
        if self.upsample:
            h = self.upsample(h)
        return h
    
class MiddleBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim=None, dropout=0.1):
        super().__init__()
        self.res_block1 = ResnetBlock(in_channels, in_channels, time_emb_dim, dropout)
        self.attention1 = AttentionBlock(in_channels)
        self.res_block2 = ResnetBlock(in_channels, in_channels, time_emb_dim, dropout)

    def forward(self, x, temb=None):
        h = x
        h = self.res_block1(h, temb)
        h = self.attention1(h)
        h = self.res_block2(h, temb)
        return h
    
class ConditionalUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, conditional_channels, time_emb_dim=None, has_attn=False, dropout_rate=0.1, upsample=True):
        super().__init__()
        self.has_attn = has_attn
        self.upsample = upsample

        res1_in_channels = in_channels + skip_channels + conditional_channels
        res2_in_channels = out_channels + skip_channels + conditional_channels

        self.res_block1 = ResnetBlock(res1_in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention1 = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.res_block2 = ResnetBlock(res2_in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention2 = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.upsample = Upsample(out_channels, with_conv=True) if upsample else None

    def forward(self, x, skips, conditionals, temb):
        x = torch.cat([x, skips[0], conditionals[0]], dim=1)
        h = self.res_block1(x, temb)
        h = self.attention1(h)
        h = torch.cat([h, skips[1], conditionals[1]], dim=1)
        h = self.res_block2(h, temb)
        h = self.attention2(h)
        if self.upsample:
            h = self.upsample(h)
        return h
    
class ControlNetPACAUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, time_emb_dim=None, has_attn=False, dropout_rate=0.1, upsample=True):
        super().__init__()
        self.has_attn = has_attn
        self.upsample = upsample
        res1_in_channels = in_channels + skip_channels
        res2_in_channels = out_channels + skip_channels
        
        self.res_block1 = ResnetBlock(res1_in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention1 = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.paca1 = PACALayer(out_channels, skip_channels, num_heads=8)
        self.res_block2 = ResnetBlock(res2_in_channels, out_channels, time_emb_dim, dropout_rate)
        self.attention2 = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.paca2 = PACALayer(out_channels, skip_channels, num_heads=8)
        self.upsample = Upsample(out_channels, with_conv=True) if upsample else None

    def forward(self, x, skips, pacas=None, temb=None):
        h = torch.cat([x, skips[0]], dim=1)
        h = self.res_block1(h, temb)
        h = self.attention1(h)
        if pacas is not None:
            h = self.paca1(h, pacas[0])

        h = torch.cat([h, skips[1]], dim=1)
        h = self.res_block2(h, temb)
        h = self.attention2(h)
        if pacas is not None:
            h = self.paca2(h, pacas[1])
        
        if self.upsample:
            h = self.upsample(h)
        return h

class PACALayer(nn.Module): # TODO: Look into the transformer cross attention used in PASD instead of nn.multihead
    def __init__(self, query_dim, kv_dim = None, num_heads=8):
        super().__init__()
        if kv_dim is None:
            kv_dim = query_dim
        self.num_heads = num_heads

        self.norm_q = Normalize(query_dim)
        self.norm_k = Normalize(kv_dim)

        self.to_q = nn.Conv2d(query_dim, query_dim, kernel_size=1)
        self.to_k = nn.Conv2d(kv_dim, query_dim, kernel_size=1)
        self.to_v = nn.Conv2d(kv_dim, query_dim, kernel_size=1)

        self.mha = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.to_out = nn.Conv2d(query_dim, query_dim, kernel_size=1)

    def forward(self, q_features, kv_features):
        residual = q_features
        b, c_q, h, w = q_features.shape
        _, _, hk, wk = kv_features.shape
        if (hk, wk) != (h, w):
            print("Warning: PACA layer input features have different spatial dimensions. This may cause issues.")

        q_features = self.norm_q(q_features)
        kv_features = self.norm_k(kv_features)

        q = self.to_q(q_features)
        k = self.to_k(kv_features)
        v = self.to_v(kv_features)

        q = q.view(b, c_q, h * w).transpose(1, 2)
        k = k.view(b, c_q, h * w).transpose(1, 2)
        v = v.view(b, c_q, h * w).transpose(1, 2)

        attn_output, _ = self.mha(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, c_q, h, w)
        attn_output = self.to_out(attn_output)

        return attn_output + residual
    
class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv.weight.data.zero_()
        if self.conv.bias is not None:
             self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_channels=64, time_emb_dim=None, dropout_rate=0.0, has_attn=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim, has_attn, dropout_rate)
        self.down2 = DownBlock(base_channels, base_channels*2, time_emb_dim, has_attn, dropout_rate) 
        self.down3 = DownBlock(base_channels*2, base_channels*4, time_emb_dim, has_attn, dropout_rate, downsample=False) 

        self.middle = MiddleBlock(base_channels*4, time_emb_dim, dropout_rate)

        self.norm_out = Normalize(base_channels*4)
        self.conv_out = nn.Conv2d(base_channels*4, out_channels*2, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x) 
        h, _ = self.down1(h)
        h, _ = self.down2(h)
        h, _ = self.down3(h)

        h = self.middle(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, dropout_rate=0.0, has_attn=False, tahn_out=True):
        super().__init__()
        self.tahn_out = tahn_out

        self.conv_in = nn.Conv2d(in_channels, base_channels * 4, kernel_size=3, padding=1)

        self.middle = MiddleBlock(base_channels * 4, None, dropout_rate)

        self.up3 = UpBlock(base_channels*4, base_channels*2, None, None, has_attn, dropout_rate)
        self.up2 = UpBlock(base_channels*2, base_channels, None, None, has_attn, dropout_rate)
        self.up1 = UpBlock(base_channels, base_channels, None, None, has_attn, dropout_rate, upsample=False)

        self.norm_out = Normalize(base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x) 
        h = self.middle(h)

        h = self.up3(h)
        h = self.up2(h)
        h = self.up1(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if self.tahn_out:
            h = torch.tanh(h)
        return h