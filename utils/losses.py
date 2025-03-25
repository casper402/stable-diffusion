import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg19(pretrained=True).features[:15].eval()

        for param in vgg.parameters():
            param.requires_grad = False

        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)

        self.vgg = vgg

    def forward(self, x, y):
        y = y.repeat(1, 3, 1, 1)
        x = x.repeat(1, 3, 1, 1)

        loss = 0.0
        for layer in self.vgg:
            x = layer(x)
            y = layer(y)
            loss += F.mse_loss(x, y)
   
        return loss
    
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.gaussian_blur = transforms.GaussianBlur(window_size, sigma)
        self.c1 = 0.01 ** 2  # Stability constant
        self.c2 = 0.03 ** 2

    def forward(self, x, y):
        # Inputs are [-1, 1] and need to be [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        # Compute local means
        mu_x = self.gaussian_blur(x)
        mu_y = self.gaussian_blur(y)

        # Compute local variances and covariances
        sigma_x = torch.clamp(self.gaussian_blur(x * x) - mu_x**2, min=1e-6)
        sigma_y = torch.clamp(self.gaussian_blur(y * y) - mu_y**2, min=1e-6)
        sigma_xy = self.gaussian_blur(x * y) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / ((mu_x**2 + mu_y**2 + self.c1) * (sigma_x + sigma_y + self.c2))
        loss = 1 - ssim_map.mean()

        return loss