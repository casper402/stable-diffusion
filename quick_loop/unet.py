import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion import Diffusion
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, UpBlock

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=128, 
                 dropout_rate=0.0):
        super().__init__()
        time_emb_dim = base_channels * 4

        ch1 = base_channels * 1
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4

        attn_res_64 = False
        attn_res_32 = True
        attn_res_16 = True
        attn_res_8 = True

        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)

        self.down1 = DownBlock(ch1, ch1, time_emb_dim, attn_res_64, dropout_rate)
        self.down2 = DownBlock(ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.down3 = DownBlock(ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.down4 = DownBlock(ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, downsample=False)

        self.middle = MiddleBlock(ch4, time_emb_dim, dropout_rate)

        self.up4 = UpBlock(ch4, ch3, ch4, time_emb_dim, attn_res_8, dropout_rate)
        self.up3 = UpBlock(ch3, ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.up2 = UpBlock(ch2, ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.up1 = UpBlock(ch1, ch1, ch1, time_emb_dim, attn_res_64, dropout_rate, upsample=False)

        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.init_conv(x)         
        x, intermediates1 = self.down1(x, t_emb)
        x, intermediates2 = self.down2(x, t_emb)
        x, intermediates3 = self.down3(x, t_emb)
        x, intermediates4 = self.down4(x, t_emb)

        x = self.middle(x, t_emb)

        x = self.up4(x, intermediates4, t_emb)
        x = self.up3(x, intermediates3, t_emb)
        x = self.up2(x, intermediates2, t_emb)
        x = self.up1(x, intermediates1, t_emb)

        x = self.final_norm(x)
        x = nonlinearity(x)
        x = self.final_conv(x)
        return x

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)
    
def load_unet(save_path=None, trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet().to(device)
    if save_path is None:
        print("UNET initialized with random weights.")
        return unet
    if os.path.exists(save_path):
        unet.load_state_dict(torch.load(save_path, map_location=device), strict=True)
        print(f"UNET loaded from {save_path}")
    else:
        print(f"UNET not found at {save_path}.")
    if not trainable:
        for param in unet.parameters():
            param.requires_grad = False
    unet.eval()
    return unet

def predict_unet(unet, vae, x_batch, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(device)
    unet.eval()
    vae.eval()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        x_batch = x_batch.to(device)
        z, _, _, x_recon_batch = vae(x_batch)
        t = diffusion.sample_timesteps(x_batch.shape[0])
        noise = torch.randn_like(z)
        z_noisy = diffusion.add_noise(z, t, noise=noise)
        pred_noise = unet(z_noisy, t)

        # Approximate denoise latent
        alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        z_denoised_pred = (z_noisy - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t

        unet_recon_batch = vae.decode(z_denoised_pred)

        for i in range(x_batch.size(0)):
            original = x_batch[i]
            unet_recon = unet_recon_batch[i]
            x_recon = x_recon_batch[i]
            original_img = (original / 2 + 0.5).clamp(0, 1)
            unet_recon_img = (unet_recon / 2 + 0.5).clamp(0, 1)
            recon_img = (x_recon / 2 + 0.5).clamp(0, 1)

            if save_path:
                images_to_save = [original_img, unet_recon_img, recon_img]
                output_filename = os.path.join(save_path, f"{i}.png")
                torchvision.utils.save_image(
                    images_to_save,
                    output_filename,
                    nrow=len(images_to_save),
                )

def train_unet(unet, vae, train_loader, val_loader, epochs=1000, save_path='unet.pth', predict_dir=None, early_stopping=None, patience=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=5.0e-5)
    if patience is None:
        patience = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',            
        factor=0.5,            
        patience=patience,           
        threshold=1e-4,        
        verbose=True,          
        min_lr=1e-6            
    )
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        # Training
        unet.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z_mu, z_logvar = vae.encode(x)
                z = vae.reparameterize(z_mu, z_logvar)
            t = diffusion.sample_timesteps(z.size(0))
            noise = torch.randn_like(z)
            z_noisy = diffusion.add_noise(z, t, noise=noise)
            pred_noise = unet(z_noisy, t)
            loss = noise_loss(pred_noise, noise)
            loss.backward()
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
        early_stopping_counter += 1
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(unet.state_dict(), save_path)
            print(f"✅ Saved new best unet at epoch {epoch+1} with val loss {val_loss:.4f}")

        if early_stopping and early_stopping_counter >= early_stopping:
            print(f"Early stopped after {early_stopping} epochs with no improvement.")
            break

        # Save predictions
        if predict_dir and (epoch + 1) % 50 == 0:
            for x in val_loader:
                predict_dir = os.path.join(predict_dir, f"epoch_{epoch+1}")
                predict_unet(unet, vae, x, save_path=predict_dir)
                break # Only predict on the first batch