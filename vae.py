import torch 
import torch.nn as nn
import os
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from torchvision.models import vgg16
from torchvision.transforms import Normalize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features[:8].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.normalize = Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # adjust if needed

    def forward(self, recon_x, x):
        recon_x = recon_x.repeat(1, 3, 1, 1)  # grayscale -> 3-channel
        x = x.repeat(1, 3, 1, 1)

        recon_x = self.normalize(recon_x)
        x = self.normalize(x)

        feat_recon = self.vgg(recon_x)
        feat_real = self.vgg(x)

        return F.mse_loss(feat_recon, feat_real)

perceptual_loss = PerceptualLoss(device=device)


def vae_loss(recon_x, x, mu, logvar, perceptual_weight=0.1, mse_weight=1.0, kl_weight=1.0):
    mse = F.mse_loss(recon_x, x)
    perceptual = perceptual_loss(recon_x, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = mse_weight * mse + perceptual_weight * perceptual + kl_weight * kl
    return total_loss
    
class CTDataset(Dataset):
    def __init__(self, CT_path):
        self.CT_slices = self._collect_slices(CT_path)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
    def _collect_slices(self, dataset_path):
        slice_paths = []
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(subdir_path):
                for slice_name in os.listdir(subdir_path):
                    slice_path = os.path.join(subdir_path, slice_name)
                    slice_paths.append(slice_path)
        return slice_paths
    
    def __len__(self):
        return len(self.CT_slices)
    
    def __getitem__(self, idx):
        CT_path = self.CT_slices[idx]
        CT_slice = Image.open(CT_path).convert("L")
        if self.transform:
            CT_slice = self.transform(CT_slice)
        return CT_slice


class Encoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # -> 128x128
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1) # -> 64x64
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1) # -> 32x32
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) # -> 16x16


        self.mu_conv = nn.Conv2d(256, 512, 1)
        self.logvar_conv = nn.Conv2d(256, 512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        mu = self.mu_conv(x)       # [B, latent_channels, 16, 16]
        logvar = self.logvar_conv(x)
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
        self.init_conv = nn.Conv2d(512, 256, 3, padding=1)
        self.fc = nn.Linear(latent_dim * 16 * 16, 256 * 16 * 16)
        self.unflatten = nn.Unflatten(1, (256, 16, 16))
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # -> 32x32
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # -> 64x64
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # -> 128x128
        self.deconv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)    # -> 256x256
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(128)
    def forward(self, z):

        x = F.relu(self.init_conv(z))

        x = F.relu(self.res1(x))  # new residual block
        x = F.relu(self.deconv1(x))  # -> 32x32
        x = F.relu(self.res2(x))
        x = F.relu(self.deconv2(x))  # -> 64x64
        x = F.relu(self.deconv3(x))  # -> 128x128
        x = torch.sigmoid(self.deconv4(x))  # -> 256x256

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
        x_hat = self.decoder(z)
        return x_hat, mu, logvar




transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = CTDataset('../training_data/CT')
subset, _ = random_split(dataset, [500, len(dataset) - 500])

train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size - 10
test_size = 10
train_dataset, val_dataset, test_dataset = random_split(subset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)


vae = VAE(latent_dim=4).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

best_val_loss = float('inf')
save_path = 'best_vae_ct1.pth'

for epoch in range(50):
    vae.train()
    train_loss = 0

    for x in train_loader:
        x = x.to(device)
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(vae.state_dict(), save_path)
        print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")









vae.load_state_dict(torch.load(save_path))
vae.eval()

# Inference loop on test_loader
pred_dir = "./predictions/"
os.makedirs(pred_dir, exist_ok=True)

with torch.no_grad():
    for i, x in enumerate(test_loader):
        x = x.to(device)
        recon, _, _ = vae(x)

        # Save the first few predictions
        for j in range(min(x.size(0), 8)):
            original = x[j]
            reconstructed = recon[j]

            # Save originals and reconstructions as image files
            vutils.save_image(original, f"{pred_dir}/img_{i}_{j}_orig.png", normalize=True)
            vutils.save_image(reconstructed, f"{pred_dir}/img_{i}_{j}_recon.png", normalize=True)

        # Optional: stop after a few batches
        if i >= 2:
            break