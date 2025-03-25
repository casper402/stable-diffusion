import torch.nn as nn

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.trans_conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

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
    
class BottleneckBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.crossAttn = CrossAttention2D(channels, num_heads=num_heads)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x_res = x # Residual connection

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.crossAttn(x, y)
        
        x = self.conv2(x)
        x = self.relu2(x)
        return x + x_res