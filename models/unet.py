import torch
import torch.nn as nn
from models.blocks import UpsampleBlock, DecoderBlock, CrossAttention2D
from torchvision import models

class CrossAttention2D(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, y):
        """
        x: (B, C, H, W) -> Query
        y: (B, C, H, W) -> Key, Value
        """
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H*W).permute(2, 0, 1) # (B, H*W, C)
        y_reshaped = y.view(B, C, H*W).permute(2, 0, 1) # (B, H*W, C)

        attn_out, _ = self.mha(x_reshaped, y_reshaped, y_reshaped)

        x_reshaped = x_reshaped + attn_out
        x_reshaped = self.activation(x_reshaped)

        x_out = x_reshaped.permute(1, 2, 0).view(B, C, H, W)
        return x_out

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, num_heads):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False) # in_channels=3 -> in_channels, Stride=2 -> Stride=1 to avoid early downsampling
        with torch.no_grad():
            avg_weight = resnet.conv1.weight.mean(dim=1, keepdim=True) # Average over RGB channels
            new_conv1.weight = nn.Parameter(avg_weight.repeat(1, in_channels, 1, 1)) # Expand across all latent channels
        resnet.conv1 = new_conv1
        resnet.maxpool = nn.Identity() # avoid early downsampling
        resnet_layers = list(resnet.children())

        self.initial = nn.Sequential(*resnet_layers[:5])
        self.encoder1 = resnet_layers[5]
        self.encoder2 = resnet_layers[6]
        self.encoder3 = resnet_layers[7]

        self.bottleneck = CrossAttention2D(2048, num_heads)

        self.up3 = UpsampleBlock(4096, 1024)
        self.decoder3 = DecoderBlock(1024, 1024)
        self.up2 = UpsampleBlock(2048, 512)
        self.decoder2 = DecoderBlock(512, 512)
        self.up1 = UpsampleBlock(1024, 256)
        self.decoder1 = DecoderBlock(256, out_channels)

    def forward(self, x, y):
        x = self.initial(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        y = self.initial(y)
        y = self.encoder1(y)
        y = self.encoder2(y)
        y = self.encoder3(y)

        x_bottleneck = self.bottleneck(x3, y)

        x = torch.cat([x_bottleneck, x3], dim=1)
        x = self.up3(x)
        x = self.decoder3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.decoder2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.decoder1(x)
        
        return x