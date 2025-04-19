import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion import Diffusion
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, ControlNetPACAUpBlock

class UNetControlPaca(nn.Module):
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

    def forward(self, x, t, down_additional_residuals, middle_additional_residual):
        additional_down_res_1_1, additional_down_res_1_2, additional_down_res_2_1, additional_down_res_2_2, additional_down_res_3_1, additional_down_res_3_2, additional_down_res_4_1, additional_down_res_4_2 = down_additional_residuals

        t_emb = self.time_embedding(t)
        h = self.init_conv(x)         

        h, (down_res_1_1, down_res_1_2) = self.down1(h, t_emb)
        h, (down_res_2_1, down_res_2_2) = self.down2(h, t_emb)
        h, (down_res_3_1, down_res_3_2) = self.down3(h, t_emb)
        h, (down_res_4_1, down_res_4_2) = self.down4(h, t_emb)

        h = self.middle(h, t_emb)
        h = h + middle_additional_residual

        skip_4_1 = torch.cat([down_res_4_1, additional_down_res_4_1], dim=1)
        skip_4_2 = torch.cat([down_res_4_2, additional_down_res_4_2], dim=1)
        h = self.up4(h, [skip_4_1, skip_4_2], [additional_down_res_4_1, additional_down_res_4_2], t_emb)

        skip_3_1 = torch.cat([down_res_3_1, additional_down_res_3_1], dim=1)
        skip_3_2 = torch.cat([down_res_3_2, additional_down_res_3_2], dim=1)
        h = self.up3(h, [skip_3_1, skip_3_2], [additional_down_res_3_1, additional_down_res_3_2], t_emb)

        skip_2_1 = torch.cat([down_res_2_1, additional_down_res_2_1], dim=1)
        skip_2_2 = torch.cat([down_res_2_2, additional_down_res_2_2], dim=1)
        h = self.up2(h, [skip_2_1, skip_2_2], [additional_down_res_2_1, additional_down_res_2_2], t_emb)

        skip_1_1 = torch.cat([down_res_1_1, additional_down_res_1_1], dim=1)
        skip_1_2 = torch.cat([down_res_1_2, additional_down_res_1_2], dim=1)
        h = self.up1(h, [skip_1_1, skip_1_2], [additional_down_res_1_1, additional_down_res_1_2], t_emb)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h
    
def load_unet_control_paca(unet_save_path=None, paca_save_path=None, unet_trainable=False, paca_trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unetControlPACA = UNetControlPaca().to(device)

    if unet_save_path is None:
        print("UNet initialized with random weights.")
    else: 
        unet_state_dict = torch.load(unet_save_path, map_location=device)
        _, unetControlPACA_unexpected_keys = unetControlPACA.load_state_dict(unet_state_dict, strict=False)
        if unetControlPACA_unexpected_keys:
            print(f"Unexpected keys in UNetControlPACA state_dict: {unetControlPACA_unexpected_keys}")

    for param in unetControlPACA.parameters():
        param.requires_grad = unet_trainable
    
    paca_params = 0
    for name, param in unetControlPACA.named_parameters():
        if 'paca' in name.lower():
            param.requires_grad = paca_trainable
            paca_params += param.numel()

    if unet_save_path:
        if unetControlPACA.parameters() - paca_params != unet_state_dict.parameters():
            print(f"WARNING: UNetControlPACA parameters - PACA parameters should be equal to the loaded state_dict parameters.")
            print(f"Loaded state_dict parameters: {unet_state_dict.parameters()}")
            print(f"UNetControlPACA parameters: {unetControlPACA.parameters()}")
            print(f"UNetControlPACA parameters - PACA parameters: {unetControlPACA.parameters() - paca_params}")

    print(f"Total UNetControlPACA parameters: {unetControlPACA.parameters()}")
    print(f"Unet parameters: {unetControlPACA.parameters() - paca_params}, trainable: {unet_trainable}")
    print(f"PACA parameters: {paca_params} , trainable: {paca_trainable}")

    return unetControlPACA
    
