import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from utils.losses import PerceptualLoss, SsimLoss
from quick_loop.blocks import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, latent_channels=3):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, base_channels)
        self.decoder = Decoder(latent_channels, out_channels, base_channels, tahn_out=True)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return z, mu, logvar, recon
    
def vae_loss(recon, x, mu, logvar, perceptual_loss, ssim_loss, perceptual_weight=0.1, ssim_weight=0.8, mse_weight=0.0, kl_weight=0.00001, l1_weight=1.0):
    mse = F.mse_loss(recon, x)
    perceptual = perceptual_loss(recon, x)
    kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]))
    ssim_val = ssim_loss(recon, x)
    l1 = F.l1_loss(recon, x)
    total_loss = mse_weight * mse + perceptual_weight * perceptual + kl_weight * kl + ssim_val * ssim_weight + l1 * l1_weight
    return total_loss
    
def load_vae(save_path=None, trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    if save_path is None:
        print("VAE initialized with random weights.")
        return vae
    if os.path.exists(save_path):
        vae.load_state_dict(torch.load(save_path, map_location=device), strict=True)
        print(f"VAE loaded from {save_path}")
    else:
        print(f"VAE not found at {save_path}.")
    if not trainable:
        for param in vae.parameters():
            param.requires_grad = False
    vae.eval()
    return vae

def predict_vae(vae, x, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.eval()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        x = x.to(device)
        _, _, _, recon = vae(x)
    for i in range(recon.shape[0]):
        recon_image = recon[i]
        original_image = x[i]
        recon_image_processed = (recon_image / 2 + 0.5).clamp(0, 1)
        original_image_processed = (original_image / 2 + 0.5).clamp(0, 1)
        if save_path:
            images_to_save = [original_image_processed, recon_image_processed]
            output_filename = os.path.join(save_path, f"{i}.png")
            torchvision.utils.save_image(
                images_to_save,
                output_filename,
                nrow=len(images_to_save),
            )

def train_vae(
        vae, 
        train_loader, 
        val_loader, 
        epochs=1000, 
        save_path='vae.pth', 
        predict_dir=None, 
        early_stopping=None, 
        patience=None, 
        perceptual_weight=0.1, 
        ssim_weight=0.8, 
        mse_weight=0.0, 
        kl_weight=0.00001, 
        l1_weight=1.0,
        learning_rate=6.25e-6 # Used to be 5
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual_loss = PerceptualLoss(device=device)
    ssim_loss = SsimLoss()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)
    if not patience:
        patience = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',            
        factor=0.5,            
        patience=patience,           
        threshold=1e-4,        
        min_lr=1e-6            
    )
    best_val_loss = float('inf')
    max_grad_norm = 1.0
    early_stopping_counter = 0

    for epoch in range(epochs):
        # Training
        vae.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(device)
            _, mu, logvar, recon = vae(x)
            loss = vae_loss(recon, x, mu, logvar, perceptual_loss, ssim_loss, perceptual_weight, ssim_weight, mse_weight, kl_weight, l1_weight)

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
                loss = vae_loss(recon, x, mu, logvar, perceptual_loss, ssim_loss, perceptual_weight, ssim_weight, mse_weight, kl_weight, l1_weight)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        early_stopping_counter += 1
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']} ")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(vae.state_dict(), save_path)
            print(f"✅ Saved new best vae at epoch {epoch+1} with val loss {val_loss:.4f}")

        if early_stopping and early_stopping_counter >= early_stopping:
            print(f"Early stopped after {early_stopping} epochs with no improvement.")
            break

        # Save predictions
        if predict_dir and (epoch+1) % 50 == 0:
            for x in val_loader:
                predict_dir = os.path.join(predict_dir, f"epoch_{epoch+1}")
                predict_vae(vae, x, save_path=predict_dir)
                break # Only predict on the first batch