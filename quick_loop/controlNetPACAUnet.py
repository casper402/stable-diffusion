import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion import Diffusion
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, ControlNetPACAUpBlock

class ControlNetPACAUNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=124,
                 dropout_rate=0.0):
        super().__init__()
        time_emb_dim = base_channels * 4

        ch1 = base_channels * 1
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4
        ch_mid = base_channels * 4

        attn_res_64 = False
        attn_res_32 = True
        attn_res_16 = True
        attn_res_8 = True

        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(ch1, ch2, time_emb_dim, attn_res_64, dropout_rate)
        self.down2 = DownBlock(ch2, ch3, time_emb_dim, attn_res_32, dropout_rate)
        self.down3 = DownBlock(ch3, ch4, time_emb_dim, attn_res_16, dropout_rate)
        self.down4 = DownBlock(ch4, ch_mid, time_emb_dim, attn_res_8, dropout_rate, downsample=False)

        self.middle = MiddleBlock(ch_mid, time_emb_dim, dropout_rate)

        self.up4 = ControlNetPACAUpBlock(ch_mid, ch4, ch_mid, time_emb_dim, attn_res_8, dropout_rate)
        self.up3 = ControlNetPACAUpBlock(ch4, ch3, ch4, time_emb_dim, attn_res_16, dropout_rate)
        self.up2 = ControlNetPACAUpBlock(ch3, ch2, ch3, time_emb_dim, attn_res_32, dropout_rate)
        self.up1 = ControlNetPACAUpBlock(ch2, ch1, ch2, time_emb_dim, attn_res_64, dropout_rate, upsample=False)

        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, control_features):
        controlnet_down_samples, controlnet_middle_samples = control_features
        zconv_middle, zconv_skip3, zconv_skip2, zconv_skip1 = zero_conv_features
        h_middle, c3, c2, c1 = projected_features

        t_emb = self.time_embedding(t)
        h = self.init_conv(x)         

        h, skip1 = self.down1(h, t_emb)
        h, skip2 = self.down2(h, t_emb)
        h, skip3 = self.down3(h, t_emb)

        h_middle_in = h + zconv_middle
        h_middle_out = self.middle(h_middle_in, t_emb)
        h = self.paca_middle(q_features=h_middle_out, kv_features=h_middle)

        modified_skip3 = skip3 + zconv_skip3
        h_up3_out = self.up3(h, modified_skip3, t_emb)
        h = self.paca3(q_features=h_up3_out, kv_features=c3)

        modified_skip2 = skip2 + zconv_skip2
        h_up2_out = self.up2(h, modified_skip2, t_emb)
        h = self.paca2(q_features=h_up2_out, kv_features=c2)

        modified_skip1 = skip1 + zconv_skip1
        h_up1_out = self.up1(h, modified_skip1, t_emb)
        h = self.paca1(q_features=h_up1_out, kv_features=c1)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h