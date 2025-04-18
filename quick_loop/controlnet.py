import os
import torch
import torch.nn as nn
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, UpBlock, ZeroConv2d

class ControlNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=124,
                 dropout_rate=0.1):
        super().__init__()
        time_emb_dim = base_channels * 4

        ch1 = base_channels * 1
        ch2 = base_channels * 1
        ch3 = base_channels * 2
        ch4 = base_channels * 4
        ch_mid = base_channels * 4

        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)

        attn_res_64 = False
        attn_res_32 = True
        attn_res_16 = True
        attn_res_8 = True

        self.down1 = DownBlock(ch1, ch2, time_emb_dim, attn_res_64, dropout_rate)
        self.down2 = DownBlock(ch2, ch3, time_emb_dim, attn_res_32, dropout_rate)
        self.down3 = DownBlock(ch3, ch4, time_emb_dim, attn_res_16, dropout_rate)
        self.down4 = DownBlock(ch4, ch_mid, time_emb_dim, attn_res_8, dropout_rate, downsample=False)
        self.middle = MiddleBlock(ch4, time_emb_dim, dropout_rate)

        self.controlnet_down_blocks = nn.ModuleList()
        num_skips_per_block = 3
        for _ in range(num_skips_per_block):
            self.controlnet_down_blocks.append(ZeroConv2d(ch2, ch2))
        for _ in range(num_skips_per_block):
            self.controlnet_down_blocks.append(ZeroConv2d(ch3, ch3))
        for _ in range(num_skips_per_block):
            self.controlnet_down_blocks.append(ZeroConv2d(ch4, ch4))
        for _ in range(num_skips_per_block):
            self.controlnet_down_blocks.append(ZeroConv2d(ch_mid, ch_mid))

        self.controlnet_middle_block = ZeroConv2d(ch_mid, ch_mid)      
    
    def forward(self, x, cond, t=None):
        h = self.time_embedding(t)
        h = self.init_conv(x)
        h = h + cond

        all_intermediate_outputs = ()

        h, intermediates1 = self.down1(h)
        all_intermediate_outputs += intermediates1
        h, intermediates2 = self.down2(h)
        all_intermediate_outputs += intermediates2
        h, intermediates3 = self.down3(h)
        all_intermediate_outputs += intermediates3
        h, intermediates4 = self.down4(h)
        all_intermediate_outputs += intermediates4

        h_middle = self.middle(h)

        controlnet_processed_down_samples = []
        if len(all_intermediate_outputs) != len(self.controlnet_down_blocks):
            raise ValueError(f"Expected {len(self.controlnet_down_blocks)} down blocks, but got {len(all_intermediate_outputs)}")
        
        for res_sample, controlnet_block in zip(all_intermediate_outputs, self.controlnet_down_blocks):
            controlnet_processed_down_samples.append(controlnet_block(res_sample))

        controlnet_processed_middle_sample = self.controlnet_middle_block(h_middle)

        return controlnet_processed_down_samples, controlnet_processed_middle_sample
