import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from quick_loop.blocks import ZeroConv2d

class DegradationRemoval(nn.Module):
    def __init__(self, condition_channels=1, final_embedding_channels=256):
        super().__init__()
        ch1 = 16
        ch2 = 32
        ch3 = 96

        self.init_conv = nn.Conv2d(condition_channels, ch1, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(ch1, ch1, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, stride=2)
        self.to_grayscale_128 = nn.Conv2d(ch2, 1, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(ch2, ch2, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(ch2, ch3, kernel_size=3, padding=1, stride=2)
        self.to_grayscale_64 = nn.Conv2d(ch3, 1, kernel_size=3, padding=1)

        self.conv_out = ZeroConv2d(ch3, final_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.init_conv(conditioning)
        embedding = F.silu(embedding)

        embedding = self.conv1(embedding)
        embedding = F.silu(embedding)
        embedding = self.down1(embedding)
        embedding = F.silu(embedding)
        pred_128 = self.to_grayscale_128(embedding) # Prediction for loss

        embedding = self.conv2(embedding)
        embedding = F.silu(embedding)
        embedding = self.down2(embedding)
        embedding = F.silu(embedding)
        pred_64 = self.to_grayscale_64(embedding) # Prediction for loss

        intermediate_preds = (pred_128, pred_64)
        embedding = self.conv_out(embedding)

        return embedding, intermediate_preds
    
def degradation_loss(intermediate_preds, ground_truth):
    pred_128, pred_64 = intermediate_preds
    gt_128 = F.interpolate(ground_truth, size=(128, 128), mode='area')
    gt_64 = F.interpolate(ground_truth, size=(64, 64), mode='area')
    loss_128 = F.l1_loss(pred_128, gt_128)
    loss_64 = F.l1_loss(pred_64, gt_64)
    return loss_128 + loss_64

def load_degradation_removal(save_path=None, trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    degradation_removal = DegradationRemoval().to(device)
    if save_path is None:
        print("Degradation Removal initialized with random weights.")
        return degradation_removal
    if os.path.exists(save_path):
        degradation_removal.load_state_dict(torch.load(save_path, map_location=device), strict=True)
        print(f"Degradation Removal loaded from {save_path}")
    else:
        print(f"Degradation Removal not found at {save_path}.")
    if not trainable:
        for param in degradation_removal.parameters():
            param.requires_grad = False
    degradation_removal.eval()
    return degradation_removal