import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Encoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)  # -> 128x128
        self.res1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # -> 64x64
        self.res2 = ResidualBlock(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # -> 32x32
        self.res3 = ResidualBlock(256)

        self.conv_mu = nn.Conv2d(256, latent_dim, 1)
        self.conv_logvar = nn.Conv2d(256, latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.res1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.res3(x))
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.deconv_input = nn.ConvTranspose2d(latent_dim, 256, 3, 1, 1)
        self.res1 = ResidualBlock(256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # -> 64x64
        self.res2 = ResidualBlock(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # -> 128x128
        self.res3 = ResidualBlock(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1, 4, 2, 1)    # -> 256x256
    
    def forward(self, z):
        x = F.relu(self.deconv_input(z))
        x = F.relu(self.res1(x))
        x = F.relu(self.deconv1(x))  # -> 64x64
        x = F.relu(self.res2(x))
        x = F.relu(self.deconv2(x))  # -> 128x128
        x = F.relu(self.res3(x))
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
