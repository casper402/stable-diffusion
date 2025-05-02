import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion import Diffusion
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, UpBlock

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        h = self.init_conv(x)         
        h, intermediates1 = self.down1(h, t_emb)
        h, intermediates2 = self.down2(h, t_emb)
        h, intermediates3 = self.down3(h, t_emb)
        h, intermediates4 = self.down4(h, t_emb)

        h = self.middle(h, t_emb)

        h = self.up4(h, intermediates4, t_emb)
        h = self.up3(h, intermediates3, t_emb)
        h = self.up2(h, intermediates2, t_emb)
        h = self.up1(h, intermediates1, t_emb)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)
    
def load_unet(save_path=None, trainable=False, base_channels=None, dropout_rate=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if base_channels is None:
        unet = UNet(dropout_rate=dropout_rate).to(device)
    else:
        unet = UNet(base_channels=base_channels, dropout_rate=dropout_rate).to(device)
        print("UNET base channels:", base_channels)
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

def predict_unet(unet, vae, x_batch, batch_idx, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(device)
    unet.eval()
    vae.eval()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        x_batch = x_batch.to(device)
        z, _, _, _ = vae(x_batch)
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
        noisy_batch = vae.decode(z_noisy)

        for i in range(x_batch.size(0)):
            original = x_batch[i]
            unet_recon = unet_recon_batch[i]
            x_noisy = noisy_batch[i]
            original_img = (original / 2 + 0.5).clamp(0, 1)
            unet_recon_img = (unet_recon / 2 + 0.5).clamp(0, 1)
            recon_img = (x_noisy / 2 + 0.5).clamp(0, 1)
            timestep = t[i].item()

            if save_path:
                images_to_save = [original_img, unet_recon_img, recon_img]
                output_filename = os.path.join(save_path, f"batch_{batch_idx}_img_{i}_t_{timestep}.png")
                torchvision.utils.save_image(
                    images_to_save,
                    output_filename,
                    nrow=len(images_to_save),
                )

def augment_with_noise(x, noise_std=0.05):
    """
    Additive Gaussian noise augmentation.
    x: Tensor of shape (B, C, H, W), assumed in [-1, 1] or [0, 1].
    noise_std: standard deviation of the noise.
    """
    noise = torch.randn_like(x) * noise_std
    x_noisy = x + noise
    # if your inputs are normalized to [-1,1], clamp there; if [0,1], clamp to [0,1]
    return x_noisy.clamp(-1.0, 1.0)

def train_unet(
    unet, 
    vae, 
    train_loader, 
    val_loader,
    test_loader, 
    epochs=1000, 
    save_path='unet.pth', 
    predict_dir=None, 
    early_stopping=None, 
    patience=None, 
    epochs_between_prediction=50,
    learning_rate=5.0e-5,
    weight_decay_val=1e-4,
    gradient_clip_val=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay_val,
    )
    if patience is None:
        patience = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',            
        factor=0.5,            
        patience=patience,           
        threshold=1e-4,        
        min_lr=1e-6            
    )
    diffusion = Diffusion(device)

    # --- Training loop ---
    best_val_loss = float('inf')
    early_stopping_counter = 0


    for epoch in range(epochs):
        unet.train()
        train_loss = 0

        for i, x in enumerate(train_loader):
            x = x.to(device)
            
            optimizer.zero_grad()

            with torch.no_grad():
                z_mu, z_logvar = vae.encode(x)
                z = vae.reparameterize(z_mu, z_logvar)
            
            # Forward pass
            t = diffusion.sample_timesteps(z.size(0))
            noise = torch.randn_like(z)
            z_noisy = diffusion.add_noise(z, t, noise=noise)
            pred_noise = unet(z_noisy, t)

            # Compute Loss
            loss = noise_loss(pred_noise, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=gradient_clip_val)

            optimizer.step()
            
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
        if predict_dir and (epoch + 1) % epochs_between_prediction == 0:
            for i , x in enumerate(test_loader):
                predict_unet(unet, vae, x, i, save_path=os.path.join(predict_dir, f"epoch_{epoch+1}"))

def train_joint(
    unet,
    vae,
    train_loader,
    val_loader,
    test_loader=None,
    epochs=1000,
    save_unet_path='unet.pth',
    save_vae_path='vae.pth',
    learning_rate=5e-6,
    weight_decay=1e-4,
    gradient_clip_val=1.0,
    early_stopping=None,
    vae_loss_weights=None,      # dict with keys: perceptual, ssim, mse, kl, l1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = AdamW(
        list(unet.parameters()) + list(vae.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    diffusion = Diffusion(device)

    best_val_loss = float('inf')
    early_counter = 0

    # unpack vae‐loss weights
    perceptual_w = vae_loss_weights.get('perceptual', 0.1)
    ssim_w       = vae_loss_weights.get('ssim',       0.8)
    mse_w        = vae_loss_weights.get('mse',        0.0)
    kl_w         = vae_loss_weights.get('kl',         1e-5)
    l1_w         = vae_loss_weights.get('l1',         1.0)

    print("starting joint training")

    for epoch in range(1, epochs+1):
        unet.train()
        vae.train()

        # running sums for this epoch
        sum_unet_loss = 0.0
        sum_vae_loss  = 0.0
        sum_combined  = 0.0

        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            # encode
            z_mu, z_logvar = vae.encode(x)
            z = vae.reparameterize(z_mu, z_logvar)

            # UNet path
            t = diffusion.sample_timesteps(z.size(0))
            noise = torch.randn_like(z)
            z_noisy = diffusion.add_noise(z, t, noise=noise)
            pred_noise = unet(z_noisy, t)
            loss_unet = F.mse_loss(pred_noise, noise)

            # VAE path
            recon = vae.decode(z)
            loss_vae = vae_loss(
                recon, x, z_mu, z_logvar,
                perceptual_loss, ssim_loss,
                perceptual_w, ssim_w, mse_w, kl_w, l1_w
            )

            # combine & step
            loss = loss_unet + loss_vae
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(vae.parameters()),
                max_norm=gradient_clip_val
            )
            optimizer.step()

            # accumulate
            sum_unet_loss += loss_unet.item()
            sum_vae_loss  += loss_vae.item()
            sum_combined  += loss.item()

        # average training losses
        n_batches = len(train_loader)
        avg_unet_train = sum_unet_loss / n_batches
        avg_vae_train  = sum_vae_loss  / n_batches
        avg_comb_train = sum_combined  / n_batches

        # validation
        unet.eval()
        vae.eval()
        sum_unet_val = 0.0
        sum_vae_val  = 0.0
        sum_comb_val = 0.0

        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)

                # unet
                t = diffusion.sample_timesteps(z.size(0))
                noise = torch.randn_like(z)
                pred = unet(diffusion.add_noise(z, t, noise), t)
                l_unet = F.mse_loss(pred, noise)

                # vae
                recon = vae.decode(z)
                l_vae = vae_loss(
                    recon, x, mu, logvar,
                    perceptual_loss, ssim_loss,
                    perceptual_w, ssim_w, mse_w, kl_w, l1_w
                )

                sum_unet_val += l_unet.item()
                sum_vae_val  += l_vae.item()
                sum_comb_val += (l_unet + l_vae).item()

        n_val = len(val_loader)
        avg_unet_val = sum_unet_val / n_val
        avg_vae_val  = sum_vae_val  / n_val
        avg_comb_val = sum_comb_val / n_val

        # logging
        print(
            f"[Epoch {epoch}] "
            f"Train UNet: {avg_unet_train:.4f}, VAE: {avg_vae_train:.4f}, Combined: {avg_comb_train:.4f}  |  "
            f"Val UNet:   {avg_unet_val:.4f}, VAE:   {avg_vae_val:.4f}, Combined: {avg_comb_val:.4f}"
        )

        # save & early stop
        if avg_comb_val < best_val_loss:
            best_val_loss = avg_comb_val
            early_counter = 0
            torch.save(unet.state_dict(), save_unet_path)
            torch.save(vae.state_dict(), save_vae_path)
            print(f"✔︎ Saved best models at epoch {epoch} (val_combined={avg_comb_val:.4f})")
        else:
            early_counter += 1
            if early_stopping and early_counter >= early_stopping:
                print(f"➡︎ Early stopping after {early_stopping} epochs with no improvement.")
                break