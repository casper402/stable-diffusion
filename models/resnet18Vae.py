import torch
import torch.nn as nn
from torchvision import models

class VAE(nn.Module):
    def __init__(self, latent_dim, in_channels=1, out_channels=1):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        with torch.no_grad():
            avg_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.weight = nn.Parameter(avg_weight.repeat(1, in_channels, 1, 1))
        resnet.conv1 = new_conv1
        resnet.maxpool = nn.Identity()
        resnet_layers = list(resnet.children())

        self.encoder = nn.Sequential(
            *resnet_layers[:8],
            nn.Conv2d(512, 2*latent_dim, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Refinement conv
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent = self.encoder(x)

        mu, logvar = torch.chunk(latent, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)
        return z, mu, logvar, reconstructed