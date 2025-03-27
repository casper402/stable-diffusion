import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)  # -> 128x128
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # -> 64x64
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # -> 32x32

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 32 * 32, latent_dim * 32 * 32)
        self.fc_logvar = nn.Linear(256 * 32 * 32, latent_dim * 32 * 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(latent_dim * 32 * 32, 256 * 32 * 32)
        self.unflatten = nn.Unflatten(1, (256, 32, 32))
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # -> 64x64
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # -> 128x128
        self.deconv3 = nn.ConvTranspose2d(32, 1, 4, 2, 1)    # -> 256x256
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(128)
    
    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)

        x = F.relu(self.res1(x))  # new residual block
        x = F.relu(self.deconv1(x))  # -> 64x64
        x = F.relu(self.res2(x))
        x = F.relu(self.deconv2(x))  # -> 128x128
        x = torch.sigmoid(self.deconv3(x))  # -> 256x256

        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return z, mu, logvar, recon
