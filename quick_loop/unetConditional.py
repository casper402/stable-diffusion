import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion import Diffusion
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, ConditionalUpBlock, UpBlock, PACALayer

class UNetSkip(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=256, 
                 dropout_rate=0.1):
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

        self.up4 = ConditionalUpBlock(ch4, ch3, ch4, ch4, time_emb_dim, attn_res_8, dropout_rate)
        self.up3 = ConditionalUpBlock(ch3, ch2, ch3, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.up2 = ConditionalUpBlock(ch2, ch1, ch2, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.up1 = ConditionalUpBlock(ch1, ch1, ch1, ch1, time_emb_dim, attn_res_64, dropout_rate, upsample=False)
        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

        # Conditioning Encoder Path
        self.cond_init_conv = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)
        self.cond_down1 = DownBlock(ch1, ch1, time_emb_dim, attn_res_64, dropout_rate)
        self.cond_down2 = DownBlock(ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.cond_down3 = DownBlock(ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.cond_down4 = DownBlock(ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, downsample=False)

    def forward(self, x, condition, t):
        t_emb = self.time_embedding(t)

        c = self.cond_init_conv(condition)
        c, cond_intermediates1 = self.cond_down1(c)
        c, cond_intermediates2 = self.cond_down2(c)
        c, cond_intermediates3 = self.cond_down3(c)
        c, cond_intermediates4 = self.cond_down4(c)

        h = self.init_conv(x)         
        h, intermediates1 = self.down1(h, t_emb)
        h, intermediates2 = self.down2(h, t_emb)
        h, intermediates3 = self.down3(h, t_emb)
        h, intermediates4 = self.down4(h, t_emb)

        h = self.middle(h, t_emb)

        h = self.up4(h, intermediates4, cond_intermediates4, t_emb)
        h = self.up3(h, intermediates3, cond_intermediates3, t_emb)
        h = self.up2(h, intermediates2, cond_intermediates2, t_emb)
        h = self.up1(h, intermediates1, cond_intermediates1, t_emb)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h
    
class UNetCrossAttention(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=256, 
                 dropout_rate=0.1):
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
        self.down1 = DownBlock(ch1, ch1, time_emb_dim, attn_res_64, dropout_rate, cross_attention=True)
        self.down2 = DownBlock(ch1, ch2, time_emb_dim, attn_res_32, dropout_rate, cross_attention=True)
        self.down3 = DownBlock(ch2, ch3, time_emb_dim, attn_res_16, dropout_rate, cross_attention=True)
        self.down4 = DownBlock(ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, downsample=False, cross_attention=True)

        self.middle = MiddleBlock(ch4, time_emb_dim, dropout_rate, cross_attention=True)

        self.up4 = UpBlock(ch4, ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, cross_attention=True)
        self.up3 = UpBlock(ch3, ch2, ch3, time_emb_dim, attn_res_16, dropout_rate, cross_attention=True)
        self.up2 = UpBlock(ch2, ch1, ch2, time_emb_dim, attn_res_32, dropout_rate, cross_attention=True)
        self.up1 = UpBlock(ch1, ch1, ch1, time_emb_dim, attn_res_64, dropout_rate, upsample=False, cross_attention=True)
        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

        self.cond_init_conv = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)
        self.cond_down1 = DownBlock(ch1, ch1, time_emb_dim, attn_res_64, dropout_rate)
        self.cond_down2 = DownBlock(ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.cond_down3 = DownBlock(ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.cond_down4 = DownBlock(ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, downsample=False)
        self.cond_middle = MiddleBlock(ch4, time_emb_dim, dropout_rate)


    def forward(self, x, condition, t):
        t_emb = self.time_embedding(t)

        c = self.cond_init_conv(condition)
        c, cond_intermediates1 = self.cond_down1(c)
        c, cond_intermediates2 = self.cond_down2(c)
        c, cond_intermediates3 = self.cond_down3(c)
        c, cond_intermediates4 = self.cond_down4(c)
        cond_middle = self.cond_middle(c)

        h = self.init_conv(x)         
        h, intermediates1 = self.down1(h, t_emb, cond_intermediates1[1])
        h, intermediates2 = self.down2(h, t_emb, cond_intermediates2[1])
        h, intermediates3 = self.down3(h, t_emb, cond_intermediates3[1])
        h, intermediates4 = self.down4(h, t_emb, cond_intermediates4[1])

        h = self.middle(h, t_emb, cond_middle)

        h = self.up4(h, intermediates4, t_emb, cond_intermediates4[1])
        h = self.up3(h, intermediates3, t_emb, cond_intermediates3[1])
        h = self.up2(h, intermediates2, t_emb, cond_intermediates2[1])
        h = self.up1(h, intermediates1, t_emb, cond_intermediates1[1])

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)
    
def load_cond_unet(save_path=None, trainable=False, base_channels=256, dropout_rate=0.1, unet_type=UNetSkip):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = unet_type(base_channels=base_channels, dropout_rate=dropout_rate).to(device)
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

def predict_cond_unet(unet, vae, ct_batch, cbct_batch, batch_idx, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(device)
    unet.eval()
    vae.eval()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        ct_batch = ct_batch.to(device)
        cbct_batch = cbct_batch.to(device)
        ct_z, _, _, _ = vae(ct_batch)
        cbct_z, _, _, _ = vae(cbct_batch)
        t = diffusion.sample_timesteps(ct_batch.shape[0])
        noise = torch.randn_like(ct_z)
        ct_z_noisy = diffusion.add_noise(ct_z, t, noise=noise)
        pred_noise = unet(ct_z_noisy, cbct_z, t)

        # Approximate denoise latent
        alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        z_denoised_pred = (ct_z_noisy - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t

        unet_recon_batch = vae.decode(z_denoised_pred)
        noisy_batch = vae.decode(ct_z_noisy)

        for i in range(cbct_batch.size(0)):
            original = ct_batch[i]
            unet_recon = unet_recon_batch[i]
            ct_noisy = noisy_batch[i]
            original_img = (original / 2 + 0.5).clamp(0, 1)
            unet_recon_img = (unet_recon / 2 + 0.5).clamp(0, 1)
            recon_img = (ct_noisy / 2 + 0.5).clamp(0, 1)
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

def train_cond_unet(
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
    learning_rate=5e-5,
    weight_decay_val=1e-4,
    gradient_clip_val=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay_val
    )
    if patience is None:
        patience = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',            
        factor=0.5,            
        patience=patience,           
        threshold=1e-4,        
        min_lr=1e-6, 
    )
    diffusion = Diffusion(device)

    # --- Training loop ---
    best_val_loss = float('inf')
    early_stopping_counter = 0

    optimizer.zero_grad()

    for epoch in range(epochs):
        unet.train()
        train_loss = 0

        for i, (ct, cbct) in enumerate(train_loader):
            ct = ct.to(device)
            cbct = cbct.to(device)

            with torch.no_grad():
                ct_z_mu, ct_z_logvar = vae.encode(ct)
                ct_z = vae.reparameterize(ct_z_mu, ct_z_logvar)

                cbct_z_mu, cbct_z_logvar = vae.encode(cbct)
                cbct_z = vae.reparameterize(cbct_z_mu, cbct_z_logvar)
            
            # Forward pass
            t = diffusion.sample_timesteps(ct_z.size(0))
            noise = torch.randn_like(ct_z)
            ct_z_noisy = diffusion.add_noise(ct_z, t, noise=noise)
            pred_noise = unet(ct_z_noisy, cbct_z, t)

            # Compute Loss
            loss = noise_loss(pred_noise, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=gradient_clip_val)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for (ct, cbct) in val_loader:
                ct = ct.to(device)
                cbct = cbct.to(device)

                ct_z_mu, ct_z_logvar = vae.encode(ct)
                ct_z = vae.reparameterize(ct_z_mu, ct_z_logvar)

                cbct_z_mu, cbct_z_logvar = vae.encode(cbct)
                cbct_z = vae.reparameterize(cbct_z_mu, cbct_z_logvar)

                t = diffusion.sample_timesteps(ct_z.size(0))
                noise = torch.randn_like(ct_z)
                ct_z_noisy = diffusion.add_noise(ct_z, t, noise)
                pred_noise = unet(ct_z_noisy, cbct_z, t)
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
            for i , (ct, cbct) in enumerate(test_loader):
                predict_cond_unet(unet, vae, ct, cbct, i, save_path=os.path.join(predict_dir, f"epoch_{epoch+1}"))