import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
import matplotlib.pyplot as plt 

from models.vae import VAE
from models.conditional import UNet, ControlNet
from models.diffusion import Diffusion

from utils.dataset import CBCTtoCTDataset

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ct_dir = '../training_data/CT'
cbct_dir = '../training_data/CBCT'
vae_weights_path = '../pretrained_models/vae.pth'
unet_weights_path = '../pretrained_models/best_unet.pth'
controlnet_save_path = 'controlnet.pth'
unet_paca_save_path = 'unet_paca_layers.pth'

subset_size = 1000
batch_size = 8
test_batch_size = 1
learning_rate = 1e-4
epochs = 500
num_workers = 8
max_grad_norm = 1.0
test_image_count = 1

transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad((0, 64, 0, 64)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

dataset = CBCTtoCTDataset(cbct_dir, ct_dir, transform=transform)
subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size - test_image_count
train_dataset, val_dataset, test_dataset = random_split(subset, [train_size, val_size, test_image_count])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

vae = VAE().to(device)
vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

unet = UNet().to(device)
unet.load_state_dict(torch.load(unet_weights_path, map_location=device), strict=False)

for param in unet.parameters():
    param.requires_grad = False

trainable_unet_params = 0
for name, param in unet.named_parameters():
    if 'paca' in name.lower(): # Check if name contains 'paca'
        param.requires_grad = True
        trainable_unet_params += param.numel()
print(f"UNet PACA parameters set to trainable ({trainable_unet_params} parameters).")
if trainable_unet_params == 0:
    print("Warning: No PACA parameters found or unfrozen in UNetI am using the original PACA layer!")

controlnet = ControlNet(in_channels=4, base_channels=128, num_heads=16).to(device)
controlnet.train()
trainable_controlnet_params = sum(p.numel() for p in controlnet.parameters())
print(f"ControlNet instantiated ({trainable_controlnet_params} trainable parameters).")

diffusion = Diffusion(device, timesteps=1000)

params_to_train = list(controlnet.parameters()) + \
                  [p for name, p in unet.named_parameters() if p.requires_grad] # Get params that were unfrozen
print(f"Total parameters to train: {sum(p.numel() for p in params_to_train)}")

if not params_to_train:
     print("Error: No parameters selected for training. Check freezing logic.")
     exit()

optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate) # Use AdamW
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20, # Adjust patience if needed
    threshold=1e-4,
    verbose=True,
    min_lr=1e-6 # Allow lower min_lr
)

# --- Training Loop ---
best_val_loss = float('inf')

for epoch in range(epochs):
    unet.train()
    controlnet.train()
    train_loss = 0

    for cbct_img, ct_img in train_loader:
        optimizer.zero_grad()

        cbct_img = cbct_img.to(device)
        ct_img = ct_img.to(device)

        with torch.no_grad():
            ct_mu, ct_logvar = vae.encode(ct_img)
            z_ct = vae.reparameterize(ct_mu, ct_logvar)
            cbct_mu, cbct_logvar = vae.encode(cbct_img)
            z_cbct = vae.reparameterize(cbct_mu, cbct_logvar)

        t = diffusion.sample_timesteps(ct_img.size(0))
        noise = torch.randn_like(z_ct)
        z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

        control_features = controlnet(z_cbct)
        pred_noise = unet(z_noisy_ct, t, control_features)

        loss = noise_loss(pred_noise, noise)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=max_grad_norm)
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    unet.eval()
    controlnet.eval()
    val_loss = 0

    with torch.no_grad():
        for cbct_img, ct_img in val_loader:
            cbct_img = cbct_img.to(device)
            ct_img = ct_img.to(device)

            ct_mu, ct_logvar = vae.encode(ct_img)
            z_ct = vae.reparameterize(ct_mu, ct_logvar)

            cbct_mu, cbct_logvar = vae.encode(cbct_img)
            z_cbct = vae.reparameterize(cbct_mu, cbct_logvar)

            t = diffusion.sample_timesteps(ct_img.size(0))
            noise = torch.randn_like(z_ct)
            z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

            control_features = controlnet(z_cbct)
            pred_noise = unet(z_noisy_ct, t, control_features)

            loss = noise_loss(pred_noise, noise)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.1e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(controlnet.state_dict(), controlnet_save_path)
        # Save only PACA layer parameters from UNet
        paca_state_dict = {k: v for k, v in unet.state_dict().items() if 'paca' in k.lower()}
        if paca_state_dict: # Save only if PACA layers exist
             torch.save(paca_state_dict, unet_paca_save_path)
             print(f"✅ Saved new best ControlNet+PACA model at epoch {epoch+1} with val loss {val_loss:.4f}")
        else:
             print(f"✅ Saved new best ControlNet model (no PACA found/saved) at epoch {epoch+1} with val loss {val_loss:.4f}")


    if (epoch % 10 == 0) or epoch==1: # Save every 10 epochs
        print(f"--- Saving prediction for epoch {epoch+1} ---")
        pred_dir = f"./predictions_control/epoch_{epoch+1}/"
        os.makedirs(pred_dir, exist_ok=True)

        unet.eval()
        controlnet.eval()
        vae.eval()

        with torch.no_grad():
            for cbct, ct in test_loader:
                cbct = cbct.to(device)
                ct = ct.to(device)

                ct_mu, ct_logvar = vae.encode(ct)
                ct_z = vae.reparameterize(ct_mu, ct_logvar)

                cbct_mu, cbct_logvar = vae.encode(cbct)
                cbct_z = vae.reparameterize(cbct_mu, cbct_logvar)

                control_features = controlnet(cbct_z)
                z_t = torch.randn_like(ct_z)
                T = diffusion.timesteps

                for t_int in tqdm(range(T - 1, -1, -1), desc="Sampling latent"): 
                    t = torch.full((ct_z.size(0),), t_int, device=device, dtype=torch.long)
                    beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                    alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                    alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                    sqrt_reciprocal_alpha_t = torch.sqrt(1.0 / alpha_t)
                    pred_noise = unet(z_t, t, control_features)
                    model_mean_coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
                    model_mean = sqrt_reciprocal_alpha_t * (z_t - model_mean_coef2 * pred_noise)
                    if t_int > 0:
                        variance = beta_t
                        noise = torch.randn_like(z_t)
                        z_t_minus_1 = model_mean + torch.sqrt(variance) * noise
                    else:
                        z_t_minus_1 = model_mean
                    z_t = z_t_minus_1

                z_0 = z_t
                generated_image = vae.decode(z_0)
                generated_image = (generated_image / 2 + 0.5).clamp(0, 1).squeeze(0)
                x_condition_vis = (cbct / 2 + 0.5).clamp(0, 1).squeeze(0)
                x_target_vis = (ct / 2 + 0.5).clamp(0, 1).squeeze(0)

                images_to_save = [x_condition_vis, x_target_vis, generated_image]
                save_filename = f"epoch_{epoch+1}_comparison.png"
                torchvision.utils.save_image(
                    images_to_save,
                    save_filename,
                    nrow=len(images_to_save),
                )
                print(f"Saved comparison image to {save_filename}")

print("Training finished.")




