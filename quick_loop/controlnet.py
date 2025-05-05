import os
import torch
import torch.nn as nn
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, UpBlock, ZeroConv2d

class ControlNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=256,
                 dropout_rate=0.1):
        super().__init__()
        time_emb_dim = base_channels * 4

        ch1 = base_channels * 1
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4

        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)

        attn_res_64 = False
        attn_res_32 = True
        attn_res_16 = True
        attn_res_8 = True

        self.down1 = DownBlock(ch1, ch1, time_emb_dim, attn_res_64, dropout_rate)
        self.down2 = DownBlock(ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.down3 = DownBlock(ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.down4 = DownBlock(ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, downsample=False)
        self.middle = MiddleBlock(ch4, time_emb_dim, dropout_rate)

        self.controlnet_down_blocks = nn.ModuleList()

        self.controlnet_down_block_1_1 = ZeroConv2d(ch1, ch1)
        self.controlnet_down_block_1_2 = ZeroConv2d(ch1, ch1)
        self.controlnet_down_block_1_3 = ZeroConv2d(ch1, ch1)
        self.controlnet_down_block_2_1 = ZeroConv2d(ch2, ch2)
        self.controlnet_down_block_2_2 = ZeroConv2d(ch2, ch2)
        self.controlnet_down_block_2_3 = ZeroConv2d(ch2, ch2)
        self.controlnet_down_block_3_1 = ZeroConv2d(ch3, ch3)
        self.controlnet_down_block_3_2 = ZeroConv2d(ch3, ch3)
        self.controlnet_down_block_3_3 = ZeroConv2d(ch3, ch3)
        self.controlnet_down_block_4_1 = ZeroConv2d(ch4, ch4)
        self.controlnet_down_block_4_2 = ZeroConv2d(ch4, ch4)
        self.controlnet_down_block_4_3 = ZeroConv2d(ch4, ch4)
        self.controlnet_middle_block = ZeroConv2d(ch4, ch4)
    
    def forward(self, x, cond, t):
        t_emb = self.time_embedding(t)
        h = self.init_conv(x)
        h = h + cond

        h, intermediates1 = self.down1(h, t_emb)
        h, intermediates2 = self.down2(h, t_emb)
        h, intermediates3 = self.down3(h, t_emb)
        h, intermediates4 = self.down4(h, t_emb)

        h_middle = self.middle(h, t_emb)

        down_res_1_1 = self.controlnet_down_block_1_1(intermediates1[0])
        down_res_1_2 = self.controlnet_down_block_1_2(intermediates1[1])
        #down_res_1_3 = self.controlnet_down_block_1_3(intermediates1[2])
        down_res_2_1 = self.controlnet_down_block_2_1(intermediates2[0])
        down_res_2_2 = self.controlnet_down_block_2_2(intermediates2[1])
        #down_res_2_3 = self.controlnet_down_block_2_3(intermediates2[2])
        down_res_3_1 = self.controlnet_down_block_3_1(intermediates3[0])
        down_res_3_2 = self.controlnet_down_block_3_2(intermediates3[1])
        #down_res_3_3 = self.controlnet_down_block_3_3(intermediates3[2])
        down_res_4_1 = self.controlnet_down_block_4_1(intermediates4[0])
        down_res_4_2 = self.controlnet_down_block_4_2(intermediates4[1])
        #down_res_4_3 = self.controlnet_down_block_4_3(intermediates4[2])
        middle_res_sample = self.controlnet_middle_block(h_middle)

        down_res_samples = (
            down_res_1_1, down_res_1_2, 
            down_res_2_1, down_res_2_2, 
            down_res_3_1, down_res_3_2, 
            down_res_4_1, down_res_4_2
        )

        return down_res_samples, middle_res_sample

def load_controlnet(save_path=None, trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controlnet = ControlNet().to(device)
    if save_path is None:
        print("ControlNet initialized with random weights.")
        return controlnet
    if os.path.exists(save_path):
        controlnet.load_state_dict(torch.load(save_path, map_location=device), strict=False)
        print(f"ControlNet loaded from {save_path}")
        # TODO: Make some checks to see which weights are loaded
    else:
        print(f"ControlNet not found at {save_path}.")
    if not trainable:
        for param in controlnet.parameters():
            param.requires_grad = False
    controlnet.eval()
    return controlnet

