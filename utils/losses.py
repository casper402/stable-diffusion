import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torchvision.transforms import Normalize
from piq import ssim
from torchvision.models import vgg16

# import lpips

# class LPIPSLoss(nn.Module):
#     def __init__(self, net='vgg', device='cuda'):
#         super().__init__()
#         self.loss_fn = lpips.LPIPS(net=net).to(device)  # options: 'vgg', 'alex', 'squeeze'

#     def forward(self, recon_x, x):
#         # Convert grayscale to 3 channels if needed
#         recon_x = recon_x.repeat(1, 3, 1, 1)
#         x = x.repeat(1, 3, 1, 1)
#         return self.loss_fn(recon_x, x).mean()

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features[:8].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # adjust if needed

    def forward(self, recon_x, x):
        recon_x = recon_x.repeat(1, 3, 1, 1)  # grayscale -> 3-channel
        x = x.repeat(1, 3, 1, 1)

        recon_x = self.normalize(recon_x)
        x = self.normalize(x)

        feat_recon = self.vgg(recon_x)
        feat_real = self.vgg(x)

        return F.mse_loss(feat_recon, feat_real)
    
class SsimLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def normalize(self, x):
        # Normalize fomr [-1, 1] to [0, 1]
        return (x + 1) / 2.0

    def forward(self, recon, x):
        recon = self.normalize(recon)
        x = self.normalize(x)

        recon = torch.clamp(recon, 0, 1)
        x = torch.clamp(x, 0, 1)
        
        if torch.isnan(recon).any():
            print("Found NaNs in recon after normalization")
            raise Exception("oh shit, recon is nan")
        if torch.isnan(x).any():
            print("Found NaNs in x after normalization")
            raise Exception("oh shit, x is nan")

        return 1 - ssim(recon, x, data_range=1.0)