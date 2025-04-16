import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import os
from models.vae import VAE
from utils.dataset import CTDataset
from utils.losses import PerceptualLoss, SsimLoss

def vae_loss(recon, x, mu, logvar, perceptual_weight=0.1, ssim_weight=0.8, mse_weight=0.5, kl_weight=0.00001, l1_weight=0.5):
    mse = F.mse_loss(recon, x)
    perceptual = perceptual_loss(recon, x)
    kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]))
    ssim_val = ssim_loss(recon, x)
    l1 = F.l1_loss(recon, x)
    total_loss = mse_weight * mse + perceptual_weight * perceptual + kl_weight * kl + ssim_val * ssim_weight + l1 * l1_weight
    return total_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

perceptual_loss = PerceptualLoss(device=device)
ssim_loss = SsimLoss(device=device)


tensor_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.nn.functional.pad(x, (0, 0, 64, 64))),
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)),
])

train_dataset = CTDatasetNPY('../data/CT/training', transform=tensor_transform)
val_dataset = CTDatasetNPY('../data/CT/validation', transform=tensor_transform)
test_dataset = CTDatasetNPY('../data/CT/test', transform=tensor_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

epochs = 1000
vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=4e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',            # minimize reconstruction or total loss
        factor=0.5,            # LR reduction factor (usually 0.5–0.1)
        patience=50,           # epochs without improvement before reduction
        threshold=1e-4,        # significant improvement threshold
        verbose=True,          # prints LR updates to keep track
        min_lr=1e-6            # minimal LR allowed
    )

best_val_loss = float('inf')
save_path = 'best_vae_ct_2.pth'
max_grad_norm = 1.0 # For gradient clipping, TODO: might need to tune this value

for epoch in range(epochs):
    vae.train()
    train_loss = 0
    
    # Training
    for x in train_loader:
        x = x.to(device)
        _, mu, logvar, recon = vae(x)
        loss = vae_loss(recon, x, mu, logvar)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=max_grad_norm) # Added beceause of wierd loss spikes
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            _, mu, logvar, recon = vae(x)
            loss = vae_loss(recon, x, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Update learning rate
    scheduler.step(val_loss)

    # Print training and validation loss
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        conuter = 0
        torch.save(vae.state_dict(), save_path)
        print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

    if (epoch+0) % 50 == 0:
        vae.eval()
        pred_dir = f"./predictions_2/epoch_{epoch+1}/"
        os.makedirs(pred_dir, exist_ok=True)

        print("Saving predictions")

        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(device)
                _, _, _, recon = vae(x)

                for j in range(min(x.size(0), 8)):
                    original = x[j]
                    reconstructed = recon[j]

                    vutils.save_image(original, f"{pred_dir}/img_{i}_{j}_orig.png", normalize=True, value_range=(-1, 1))
                    vutils.save_image(reconstructed, f"{pred_dir}/img_{i}_{j}_recon.png", normalize=True, value_range=(-1, 1))

                if i >= 2:  # Save only a few batches
                    break

# Load best model for final evaluation
vae.load_state_dict(torch.load(save_path))
vae.eval()

# Inference loop on test_loader
pred_dir = "./predictions_2/best_val_loss"
os.makedirs(pred_dir, exist_ok=True)

with torch.no_grad():
    for i, x in enumerate(test_loader):
        x = x.to(device)
        _ , _, _, recon = vae(x)

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

    




