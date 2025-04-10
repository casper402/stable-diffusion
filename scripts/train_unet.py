import torchvision.utils as vutils
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import os
from models.vae import VAE
from models.unet import UNet
from models.diffusion import Diffusion
from utils.dataset import CTDataset

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CTDataset('../training_data/CT', transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad((0, 64, 0, 64)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]))

subset_size = 5000
subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

test_size = 10
train_size = int(0.8 * len(subset) - test_size)
val_size = len(subset) - train_size - test_size
train_dataset, val_dataset, test_dataset = random_split(subset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

vae = VAE().to(device)
vae_path = '../pretrained_models/vae.pth'
vae.load_state_dict(torch.load(vae_path, map_location=device))
vae.eval()

unet = UNet().to(device)
diffusion = Diffusion(device, timesteps=1000)

optimizer = torch.optim.AdamW(unet.parameters(), lr=5.0e-05)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=50,
    threshold=1e-4,
    verbose=True,
    min_lr=1e-7
)

epochs = 1000
best_val_loss = float('inf')
save_path = 'best_unet.pth'
max_grad_norm = 1.0 # For gradient clipping, TODO: might need to tune this value

for epoch in range(epochs):
    unet.train()
    train_loss = 0
    
    # Training
    for x in train_loader:
        x = x.to(device)
        
        with torch.no_grad():
            z_mu, z_logvar = vae.encode(x)
            z = vae.reparameterize(z_mu, z_logvar)

        t = diffusion.sample_timesteps(x.size(0))
        noise = torch.randn_like(z)
        z_noisy = diffusion.add_noise(z, t, noise=noise)

        pred_noise = unet(z_noisy, t)

        loss = noise_loss(pred_noise, noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=max_grad_norm) # Added beceause of wierd loss spikes
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    unet.eval()
    val_loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            z_mu, z_logvar = vae.encode(x)
            z = vae.reparameterize(z_mu, z_logvar)

            t = diffusion.sample_timesteps(x.size(0))
            noise = torch.randn_like(z)
            z_noisy = diffusion.add_noise(z, t, noise)
            
            pred_noise = unet(z_noisy, t)
            
            loss = noise_loss(pred_noise, noise)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        conuter = 0
        torch.save(unet.state_dict(), save_path)
        print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

    if (epoch) % 50 == 0:
        unet.eval()
        pred_dir = f"./predictions/epoch_{epoch+1}/"
        os.makedirs(pred_dir, exist_ok=True)

        print("Saving predictions")

        with torch.no_grad():
            for i, x_batch in enumerate(test_loader):
                x_batch = x_batch.to(device)
                
                z_mu, z_logvar = vae.encode(x_batch)
                z = vae.reparameterize(z_mu, z_logvar)

                t = diffusion.sample_timesteps(x_batch.size(0))
                noise = torch.randn_like(z)
                z_noisy = diffusion.add_noise(z, t, noise)

                pred_noise = unet(z_noisy, t)

                # Approximate denoised latent
                alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

                z_denoised_pred = (z_noisy - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t

                unet_recon_batch = vae.decode(z_denoised_pred)
                vae_recon_batch = vae.decode(z)

                num_images_to_save = min(x_batch.size(0), 8)

                for j in range(num_images_to_save):
                    original_img = x_batch[j]
                    vae_recon_img = vae_recon_batch[j]
                    unet_recon_img = unet_recon_batch[j]

                    vutils.save_image(original_img, f"{pred_dir}/img_{i}_{j}_orig.png", normalize=True, value_range=(-1, 1))
                    vutils.save_image(unet_recon_img, f"{pred_dir}/img_{i}_{j}_t={t[j]}_unet_recon.png", normalize=True, value_range=(-1, 1))
                    vutils.save_image(vae_recon_img, f"{pred_dir}/img_{i}_{j}_vae_recon.png", normalize=True, value_range=(-1, 1))

                if i >= 3:  # Save only a few batches
                    break

# Load best model for final evaluation
unet.load_state_dict(torch.load(save_path))
unet.eval()

pred_dir = f"./predictions/best_loss/"
os.makedirs(pred_dir, exist_ok=True)

print("Saving predictions")

with torch.no_grad():
    for i, x_batch in enumerate(test_loader):
        x_batch = x_batch.to(device)
        
        z_mu, z_logvar = vae.encode(x_batch)
        z = vae.reparameterize(z_mu, z_logvar)

        t = diffusion.sample_timesteps(x_batch.size(0))
        noise = torch.randn_like(z)
        z_noisy = diffusion.add_noise(z, t, noise)

        pred_noise = unet(z_noisy, t)

        # Approximate denoised latent
        alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

        z_denoised_pred = (z_noisy - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t

        unet_recon_batch = vae.decode(z_denoised_pred)
        vae_recon_batch = vae.decode(z_noisy)

        num_images_to_save = min(x_batch.size(0), 8)

        for j in range(num_images_to_save):
            original_img = x_batch[j]
            vae_recon_img = vae_recon_batch[j]
            unet_recon_img = unet_recon_batch[j]

            vutils.save_image(original_img, f"{pred_dir}/img_{i}_{j}_orig.png", normalize=True, value_range=(-1, 1))
            vutils.save_image(unet_recon_img, f"{pred_dir}/img_{i}_{j}_t={t[j]}_unet_recon.png", normalize=True, value_range=(-1, 1))
            vutils.save_image(vae_recon_img, f"{pred_dir}/img_{i}_{j}_vae_recon.png", normalize=True, value_range=(-1, 1))

        if i >= 3:  # Save only a few batches
            break





