"""
class ControlNetConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        return_rgbs: bool = True,
        use_rrdb: bool = True,
    ):
        super().__init__()

        self.return_rgbs = return_rgbs
        self.use_rrdb = use_rrdb

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        if self.use_rrdb:
            from basicsr.archs.rrdbnet_arch import RRDB
            num_rrdb_block = 2
            layers = (RRDB(block_out_channels[0], block_out_channels[0]) for i in range(num_rrdb_block))
            self.preprocesser = nn.Sequential(*layers)

        self.blocks = nn.ModuleList([])
        self.to_rgbs = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

            if return_rgbs:
                self.to_rgbs.append(nn.Conv2d(channel_out, 3, kernel_size=3, padding=1)) # channel_in

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        if self.use_rrdb:
            embedding = self.preprocesser(embedding)

        out_rgbs = []
        for i, block in enumerate(self.blocks):
            embedding = block(embedding)
            embedding = F.silu(embedding)

            if i%2 and self.return_rgbs: # 0
                out_rgbs.append(self.to_rgbs[i//2](embedding))

        embedding = self.conv_out(embedding)

        return [embedding, out_rgbs] if self.return_rgbs else embedding
"""

class DegradationRemoval(nn.Module):
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
        self.final_conv = nn.Conv2d(ch4, final_out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.init_conv(x)

        x, _ = self.down1(x)
        pred_128 = self.to_grayscale_128(x) # Prediction for loss

        x, _ = self.down2(x) # -> [B, ch3, 64, 64]
        pred_64 = self.to_grayscale_64(x) # Prediction for loss

        x, _ = self.down3(x) # -> [B, ch4, 32, 32]
        pred_32 = self.to_grayscale_32(x) # Prediction for loss

        x = self.final_norm(x)
        x = nonlinearity(x)
        x = self.final_conv(x) # -> [B, 4, 32, 32]

        intermediate_preds = (pred_128, pred_64, pred_32)

        return x, intermediate_preds
