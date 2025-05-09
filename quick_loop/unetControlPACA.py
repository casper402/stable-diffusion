import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from models.diffusion import Diffusion
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, ControlNetPACAUpBlock
from quick_loop.degradationRemoval import degradation_loss
from quick_loop.unet import noise_loss
from utils.losses import PerceptualLoss, SsimLoss
from torch.optim import AdamW

class UNetControlPaca(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=256,
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

        self.up4 = ControlNetPACAUpBlock(ch4, ch3, ch4, time_emb_dim, attn_res_8, dropout_rate)
        self.up3 = ControlNetPACAUpBlock(ch3, ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.up2 = ControlNetPACAUpBlock(ch2, ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.up1 = ControlNetPACAUpBlock(ch1, ch1, ch1, time_emb_dim, attn_res_64, dropout_rate, upsample=False)

        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, down_paca_control_residuals=None, middle_paca_control_residual=None, down_control_residuals=None, middle_control_residual=None):
        control_paca = True if down_paca_control_residuals is not None else False
        extra_control = True if down_control_residuals is not None else False

        if control_paca:
            additional_down_res_1_1, additional_down_res_1_2, additional_down_res_2_1, additional_down_res_2_2, additional_down_res_3_1, additional_down_res_3_2, additional_down_res_4_1, additional_down_res_4_2 = down_paca_control_residuals

        if extra_control:
            extra_additional_down_res_1_1, extra_additional_down_res_1_2, extra_additional_down_res_2_1, extra_additional_down_res_2_2, extra_additional_down_res_3_1, extra_additional_down_res_3_2, extra_additional_down_res_4_1, extra_additional_down_res_4_2 = down_control_residuals

        t_emb = self.time_embedding(t)
        h = self.init_conv(x)         

        h, (down_res_1_1, down_res_1_2) = self.down1(h, t_emb)
        h, (down_res_2_1, down_res_2_2) = self.down2(h, t_emb)
        h, (down_res_3_1, down_res_3_2) = self.down3(h, t_emb)
        h, (down_res_4_1, down_res_4_2) = self.down4(h, t_emb)

        if control_paca:
            down_res_1_1 += additional_down_res_1_1
            down_res_1_2 += additional_down_res_1_2
            down_res_2_1 += additional_down_res_2_1
            down_res_2_2 += additional_down_res_2_2
            down_res_3_1 += additional_down_res_3_1
            down_res_3_2 += additional_down_res_3_2
            down_res_4_1 += additional_down_res_4_1
            down_res_4_2 += additional_down_res_4_2

        if extra_control:
            down_res_1_1 += extra_additional_down_res_1_1
            down_res_1_2 += extra_additional_down_res_1_2
            down_res_2_1 += extra_additional_down_res_2_1
            down_res_2_2 += extra_additional_down_res_2_2
            down_res_3_1 += extra_additional_down_res_3_1
            down_res_3_2 += extra_additional_down_res_3_2
            down_res_4_1 += extra_additional_down_res_4_1
            down_res_4_2 += extra_additional_down_res_4_2

        h = self.middle(h, t_emb)

        if control_paca:
            h = h + middle_paca_control_residual
            h = self.up4(h, (down_res_4_1, down_res_4_2), (additional_down_res_4_1, additional_down_res_4_2), t_emb)
            h = self.up3(h, (down_res_3_1, down_res_3_2), (additional_down_res_3_1, additional_down_res_3_2), t_emb)
            h = self.up2(h, (down_res_2_1, down_res_2_2), (additional_down_res_2_1, additional_down_res_2_2), t_emb)
            h = self.up1(h, (down_res_1_1, down_res_1_2), (additional_down_res_1_1, additional_down_res_1_2), t_emb)
        
        if not control_paca:
            h = self.up4(h, (down_res_4_1, down_res_4_2), None, t_emb)
            h = self.up3(h, (down_res_3_1, down_res_3_2), None, t_emb)
            h = self.up2(h, (down_res_2_1, down_res_2_2), None, t_emb)
            h = self.up1(h, (down_res_1_1, down_res_1_2), None, t_emb)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h
    
def load_unet_control_paca(unet_save_path=None, paca_save_path=None, unet_trainable=False, paca_trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unetControlPACA = UNetControlPaca().to(device)

    if paca_save_path:
        paca_state_dict = torch.load(paca_save_path, map_location=device)
        _, paca_unexpected_keys = unetControlPACA.load_state_dict(paca_state_dict, strict=False)
        if paca_unexpected_keys:
            print(f"Unexpected keys in PACA state_dict: {paca_unexpected_keys}")

    if unet_save_path is None:
        print("UNet initialized with random weights.")
    else: 
        unet_state_dict = torch.load(unet_save_path, map_location=device)
        _, unetControlPACA_unexpected_keys = unetControlPACA.load_state_dict(unet_state_dict, strict=False)
        if unetControlPACA_unexpected_keys:
            print(f"Unexpected keys in UNetControlPACA state_dict: {unetControlPACA_unexpected_keys}")

    for param in unetControlPACA.parameters():
        param.requires_grad = unet_trainable
    
    paca_params = 0
    for name, param in unetControlPACA.named_parameters():
        if 'paca' in name.lower():
            param.requires_grad = paca_trainable
            paca_params += param.numel()
    
    unet_control_paca_params = sum(p.numel() for p in unetControlPACA.parameters())
    unet_params = sum(p.numel() for p in unet_state_dict.values())
    if unet_save_path:
        if unet_control_paca_params - paca_params != unet_params:
            print(f"WARNING: UNetControlPACA parameters - PACA parameters should be equal to the loaded state_dict parameters.")
            print(f"Loaded state_dict parameters: {unet_params}")
            print(f"UNetControlPACA parameters: {unet_control_paca_params}")
            print(f"UNetControlPACA parameters - PACA parameters: {unet_control_paca_params - paca_params}")
        print(f"UNetControlPACA loaded from {unet_save_path}")

    return unetControlPACA

def train_dr_control_paca(
    vae, 
    unet, 
    controlnet, 
    dr_module, 
    train_loader, 
    val_loader, 
    epochs=1000, 
    save_dir='.', 
    predict_dir="predictions", 
    early_stopping=None, 
    patience=None, 
    gamma=1.0, 
    guidance_scale=1.0, 
    epochs_between_prediction=50, 
    learning_rate=5.0e-5, 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)
    amp_enabled = torch.cuda.is_available()
    print(f"AMP Enabled: {amp_enabled}")
    vae.to(device)
    unet.to(device)
    controlnet.to(device)
    dr_module.to(device)
    
    # Collect trainable parameters per model
    controlnet_params = [p for p in controlnet.parameters() if p.requires_grad]
    dr_module_params = [p for p in dr_module.parameters() if p.requires_grad]
    unet_params = [p for name, p in unet.named_parameters() if p.requires_grad]
    params_to_train = controlnet_params + dr_module_params + unet_params

    # Count them
    controlnet_param_count = sum(p.numel() for p in controlnet_params)
    dr_module_param_count = sum(p.numel() for p in dr_module_params)
    unet_param_count = sum(p.numel() for p in unet_params)
    total_param_count = controlnet_param_count + dr_module_param_count + unet_param_count

    # Print summary
    print(f"Trainable parameters in ControlNet: {controlnet_param_count}")
    print(f"Trainable parameters in DR Module:  {dr_module_param_count}")
    print(f"Trainable parameters in UNet:       {unet_param_count}")
    print(f"Total parameters to train:          {total_param_count}")
    
    optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate) # Use AdamW
    if patience is None:
        patience = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience,
        threshold=1e-4,
        min_lr=min(1e-7, learning_rate)
    )
    diffusion = Diffusion(device, timesteps=1000)

    # --- Training Loop ---
    best_val_loss = float('inf')
    early_stopping_counter = 0

    optimizer.zero_grad()

    for epoch in range(epochs):
        unet.train()
        controlnet.train()
        dr_module.train()
        train_loss_total = 0
        train_loss_diff = 0
        train_loss_dr = 0

        for i, (ct_img, cbct_img) in enumerate(train_loader):
            cbct_img = cbct_img.to(device)
            ct_img = ct_img.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

            # Forward pass
            controlnet_input, intermediate_preds = dr_module(cbct_img)
            t = diffusion.sample_timesteps(z_ct.size(0))
            noise = torch.randn_like(z_ct)
            z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)
            down_res_samples, middle_res_sample = controlnet(z_noisy_ct, controlnet_input, t)
            pred_noise = unet(z_noisy_ct, t, down_res_samples, middle_res_sample)

            # Compute losses
            loss_dr = degradation_loss(intermediate_preds, ct_img)
            loss_diff = noise_loss(pred_noise, noise)
            total_loss = loss_diff + gamma * loss_dr

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)

            optimizer.step()

            train_loss_total += total_loss.item()
            train_loss_diff += loss_diff.item()
            train_loss_dr += loss_dr.item()
            
        avg_train_loss_total = train_loss_total / len(train_loader)
        avg_train_loss_diff = train_loss_diff / len(train_loader)
        avg_train_loss_dr = train_loss_dr / len(train_loader)

        # --- Validation Loop ---
        unet.eval()
        controlnet.eval()
        dr_module.eval()
        val_loss_total = 0
        val_loss_diff = 0
        val_loss_dr = 0

        with torch.no_grad():
            for ct_img, cbct_img in val_loader:
                cbct_img = cbct_img.to(device)
                ct_img = ct_img.to(device)

                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

                controlnet_input, intermediate_preds = dr_module(cbct_img)

                loss_dr = degradation_loss(intermediate_preds, ct_img)

                t = diffusion.sample_timesteps(z_ct.size(0))
                noise = torch.randn_like(z_ct)
                z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

                down_res_samples, middle_res_sample = controlnet(z_noisy_ct, controlnet_input, t)
                pred_noise = unet(z_noisy_ct, t, down_res_samples, middle_res_sample)

                loss_diff = noise_loss(pred_noise, noise)
                total_loss = loss_diff + gamma * loss_dr

                val_loss_total += total_loss.item()
                val_loss_diff += loss_diff.item()
                val_loss_dr += loss_dr.item()

        avg_val_loss_total = val_loss_total / len(val_loader)
        avg_val_loss_diff = val_loss_diff / len(val_loader)
        avg_val_loss_dr = val_loss_dr / len(val_loader)
        
        scheduler.step(avg_val_loss_total)
        early_stopping_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss_total:.6f} (Diff: {avg_train_loss_diff:.6f}, DR: {avg_train_loss_dr:.6f}) | "
            f"Val Loss: {avg_val_loss_total:.6f} (Diff: {avg_val_loss_diff:.6f}, DR: {avg_val_loss_dr:.6f}) | LR: {current_lr:.1e}")
        
        if avg_val_loss_total < best_val_loss:
            best_val_loss = avg_val_loss_total
            early_stopping_counter = 0
            
            torch.save(controlnet.state_dict(), os.path.join(save_dir, "controlnet.pth"))
            torch.save(dr_module.state_dict(), os.path.join(save_dir, "dr_module.pth"))
            # Save only PACA layer parameters from UNet
            paca_state_dict = {k: v for k, v in unet.state_dict().items() if 'paca' in k.lower()}
            if paca_state_dict: # Save only if PACA layers exist
                torch.save(paca_state_dict, os.path.join(save_dir, "paca_layers.pth"))
                print(f"✅ Saved new best ControlNet+PACA model at epoch {epoch+1} with val loss {avg_val_loss_total:.6f}")
            else:
                print(f"✅ Saved new best ControlNet+DR model (no PACA found/saved) at epoch {epoch+1} with val loss {avg_val_loss_total:.6f}")

        if early_stopping and early_stopping_counter >= early_stopping:
            print(f"Early stopped after {early_stopping} epochs with no improvement.")
            break

        # --- Inference/Saving Test Images ---
        if ((epoch + 1) % epochs_between_prediction == 0): # Save every 10 epochs
            print(f"--- Saving prediction for epoch {epoch+1} ---")

            unet.eval()
            controlnet.eval()
            dr_module.eval()
            vae.eval()

            num_images_to_save = 5
            saved_count = 0

            with torch.no_grad():
                for i, (ct, cbct) in enumerate(val_loader):
                    ct = ct.to(device)
                    cbct = cbct.to(device)

                    controlnet_input, _ = dr_module(cbct)

                    z_t = torch.randn_like(vae.encode(ct)[0])
                    T = diffusion.timesteps

                    for t_int in range(T - 1, -1, -1): 
                        t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                        # CFG: Predict noise twice
                        down_res_samples, middle_res_sample = controlnet(z_t, controlnet_input, t)
                        pred_noise_cond = unet(z_t, t, down_res_samples, middle_res_sample)
                        pred_noise_uncond = unet(z_t, t, None, None)
                        pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

                        # DDPM
                        beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                        alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                        alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                        sqrt_reciprocal_alpha_t = torch.sqrt(1.0 / alpha_t)

                        model_mean_coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
                        model_mean = sqrt_reciprocal_alpha_t * (z_t - model_mean_coef2 * pred_noise)

                        if t_int > 0:
                            variance = diffusion.beta[t_int].view(-1, 1, 1, 1) # Use posterior variance beta_t
                            noise = torch.randn_like(z_t)
                            z_t_minus_1 = model_mean + torch.sqrt(variance) * noise
                        else:
                            z_t_minus_1 = model_mean
                        z_t = z_t_minus_1

                    # Decode final latent
                    z_0 = z_t
                    generated_image_batch = vae.decode(z_0)

                    for j in range(generated_image_batch.size(0)):
                        generated_image = generated_image_batch[j]
                        cbct_image = cbct[j]
                        ct_image = ct[j]

                        generated_image_vis = (generated_image / 2 + 0.5).clamp(0, 1)
                        cbct_image_vis = (cbct_image / 2 + 0.5).clamp(0, 1)
                        ct_image_vis = (ct_image / 2 + 0.5).clamp(0, 1)

                        images_to_save = [cbct_image_vis, generated_image_vis, ct_image_vis]
                        save_filename = f"{predict_dir}/epoch_{epoch}_batch_{i}_img_{j}_guidance_scale_{guidance_scale}.png"

                        torchvision.utils.save_image(
                            images_to_save,
                            save_filename,
                            nrow=len(images_to_save),
                        )
                        saved_count += 1
                        if saved_count >= num_images_to_save:
                            break
                    if saved_count >= num_images_to_save:
                        break
            print(f"Saved {num_images_to_save} images for epoch {epoch+1} to {predict_dir}")

    print("Training finished.")

def train_dr_control_paca_v2(
    vae,
    unet,
    controlnet,
    dr_module,
    train_loader,
    val_loader,
    epochs=1000,
    save_dir='.',
    predict_dir="predictions",
    early_stopping=None,
    patience=None,
    gamma=1.0,
    guidance_scale=1.0,
    epochs_between_prediction=10,
    learning_rate=5.0e-5,
    # Reconstruction loss weights
    mse_weight=0.25,
    ssim_weight=0.50,
    perceptual_weight=0.01,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)
    amp_enabled = torch.cuda.is_available()
    print(f"AMP Enabled: {amp_enabled}")

    # Move models to device
    vae.to(device)
    unet.to(device)
    controlnet.to(device)
    dr_module.to(device)

    # Initialize loss functions
    ssim_loss = SsimLoss()
    perc_loss_fn = PerceptualLoss(device=device)

    # Collect trainable parameters
    controlnet_params = [p for p in controlnet.parameters() if p.requires_grad]
    dr_module_params = [p for p in dr_module.parameters() if p.requires_grad]
    unet_params = [p for _, p in unet.named_parameters() if p.requires_grad]
    params_to_train = controlnet_params + dr_module_params + unet_params

    optimizer = AdamW(params_to_train, lr=learning_rate)
    if patience is None:
        patience = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience,
        threshold=1e-4,
        min_lr=min(1e-7, learning_rate)
    )
    diffusion = Diffusion(device, timesteps=1000)

    # Tracking best validation loss
    best_val_loss = float('inf')
    es_counter = 0

    print(f"Starting training for {epochs} epochs. LR={learning_rate:.1e}")
    print(f"Loss weights -> MSE: {mse_weight}, SSIM: {ssim_weight}, Perceptual: {perceptual_weight}, Gamma: {gamma}")

    for epoch in range(epochs):
        unet.train(); controlnet.train(); dr_module.train()
        train_metrics = {k: 0.0 for k in ['noise','dr','mse','ssim','perceptual','total']}

        # Training loop
        for ct_img, cbct_img in train_loader:
            ct_img = ct_img.to(device)
            cbct_img = cbct_img.to(device)
            optimizer.zero_grad()

            # VAE encoding
            with torch.no_grad():
                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

            # Degradation and noise prediction
            controlnet_input, intermediate_preds = dr_module(cbct_img)
            t = diffusion.sample_timesteps(z_ct.size(0))
            noise = torch.randn_like(z_ct)
            z_noisy = diffusion.add_noise(z_ct, t, noise=noise)
            down_res, mid_res = controlnet(z_noisy, controlnet_input, t)
            pred_noise = unet(z_noisy, t, down_res, mid_res)

            # Compute losses
            loss_diff = noise_loss(pred_noise, noise)
            loss_dr   = degradation_loss(intermediate_preds, ct_img)

            # Reconstruct image for reconstruction losses
            alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1,1,1,1)
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
            z_hat = (z_noisy - sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha_cumprod
            x_recon = vae.decode(z_hat)

            mse_l = F.mse_loss(x_recon, ct_img)
            ssim_l = ssim_loss(x_recon, ct_img)
            perc_l = perc_loss_fn(x_recon, ct_img)

            # Compute weighted components for accurate reporting
            new_noise = loss_diff.item()
            new_dr    = loss_dr.item() * gamma
            new_mse   = mse_l.item() * mse_weight
            new_ssim  = ssim_l.item() * ssim_weight
            new_perc  = perc_l.item() * perceptual_weight

            # Total combined loss
            total_loss = (
                loss_diff
                + gamma * loss_dr
                + mse_weight * mse_l
                + ssim_weight * ssim_l
                + perceptual_weight * perc_l
            )

            # Backpropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()

            # Accumulate metrics using weighted values
            train_metrics['noise']      += new_noise
            train_metrics['dr']         += new_dr
            train_metrics['mse']        += new_mse
            train_metrics['ssim']       += new_ssim
            train_metrics['perceptual'] += new_perc
            train_metrics['total']      += (new_noise + new_dr + new_mse + new_ssim + new_perc)

        # Average training metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        # Validation loop
        unet.eval(); controlnet.eval(); dr_module.eval()
        val_metrics = {k: 0.0 for k in train_metrics}
        with torch.no_grad():
            for ct_img, cbct_img in val_loader:
                ct_img = ct_img.to(device)
                cbct_img = cbct_img.to(device)

                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

                controlnet_input, intermediate_preds = dr_module(cbct_img)
                loss_dr = degradation_loss(intermediate_preds, ct_img)

                t = diffusion.sample_timesteps(z_ct.size(0))
                noise = torch.randn_like(z_ct)
                z_noisy = diffusion.add_noise(z_ct, t, noise=noise)
                down_res, mid_res = controlnet(z_noisy, controlnet_input, t)
                pred_noise = unet(z_noisy, t, down_res, mid_res)
                loss_diff = noise_loss(pred_noise, noise)

                alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1,1,1,1)
                sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
                z_hat = (z_noisy - sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha_cumprod
                x_recon = vae.decode(z_hat)

                mse_l  = F.mse_loss(x_recon, ct_img)
                ssim_l = ssim_loss(x_recon, ct_img)
                perc_l = perc_loss_fn(x_recon, ct_img)

                # Compute weighted components
                new_noise = loss_diff.item()
                new_dr    = loss_dr.item() * gamma
                new_mse   = mse_l.item() * mse_weight
                new_ssim  = ssim_l.item() * ssim_weight
                new_perc  = perc_l.item() * perceptual_weight

                # Accumulate validation metrics
                val_metrics['noise']      += new_noise
                val_metrics['dr']         += new_dr
                val_metrics['mse']        += new_mse
                val_metrics['ssim']       += new_ssim
                val_metrics['perceptual'] += new_perc
                val_metrics['total']      += (new_noise + new_dr + new_mse + new_ssim + new_perc)

        # Average validation metrics
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)

        # Update LR and check for best
        scheduler.step(val_metrics['total'])
        print(
            f"Epoch {epoch+1} | "
            f"Train total={train_metrics['total']:.6f} (noise={train_metrics['noise']:.6f}, dr={train_metrics['dr']:.6f}, "
            f"mse={train_metrics['mse']:.6f}, ssim={train_metrics['ssim']:.6f}, perc={train_metrics['perceptual']:.6f}) | "
            f"Val total={val_metrics['total']:.6f} (noise={val_metrics['noise']:.6f}, dr={val_metrics['dr']:.6f}, "
            f"mse={val_metrics['mse']:.6f}, ssim={val_metrics['ssim']:.6f}, perc={val_metrics['perceptual']:.6f})"
        )

        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            es_counter = 0
            torch.save(controlnet.state_dict(), os.path.join(save_dir, "controlnet.pth"))
            torch.save(dr_module.state_dict(), os.path.join(save_dir, "dr_module.pth"))
            paca_sd = {k: v for k, v in unet.state_dict().items() if 'paca' in k.lower()}
            if paca_sd:
                torch.save(paca_sd, os.path.join(save_dir, "paca_layers.pth"))
            print(f"✅ Saved best model at epoch {epoch+1} (val={best_val_loss:.6f})")
        else:
            es_counter += 1
            if early_stopping and es_counter >= early_stopping:
                print(f"⏹ Early stopping after {early_stopping} epochs")
                break

        # Inference/Saving Test Images
        if (epoch + 1) % epochs_between_prediction == 0:
            print(f"--- Saving prediction for epoch {epoch+1} ---")

            unet.eval()
            controlnet.eval()
            dr_module.eval()
            vae.eval()

            num_images_to_save = 5
            saved_count = 0

            with torch.no_grad():
                for i, (ct, cbct) in enumerate(val_loader):
                    ct = ct.to(device)
                    cbct = cbct.to(device)

                    controlnet_input, _ = dr_module(cbct)

                    # Start from random noise for generation
                    z_t = torch.randn_like(vae.encode(ct)[0])
                    T = diffusion.timesteps

                    for t_int in range(T - 1, -1, -1):
                        t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                        # CFG: Conditional and unconditional noise predictions
                        down_res, mid_res = controlnet(z_t, controlnet_input, t)
                        pred_noise_cond   = unet(z_t, t, down_res, mid_res)
                        pred_noise_uncond = unet(z_t, t, None, None)
                        # Guided noise
                        pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

                        # DDPM sampling step
                        beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                        alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                        alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                        sqrt_one_minus_acp = torch.sqrt(1.0 - alpha_cumprod_t)
                        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)

                        model_mean_coef = beta_t / sqrt_one_minus_acp
                        model_mean = sqrt_recip_alpha * (z_t - model_mean_coef * pred_noise)

                        if t_int > 0:
                            noise_term = torch.randn_like(z_t)
                            z_t = model_mean + torch.sqrt(beta_t) * noise_term
                        else:
                            z_t = model_mean

                    # Decode final latent back to image
                    z_0 = z_t
                    generated_batch = vae.decode(z_0)

                    # Save a few samples
                    for j in range(generated_batch.size(0)):
                        gen_img = generated_batch[j]
                        cbct_vis = (cbct[j] / 2 + 0.5).clamp(0,1)
                        gen_vis  = (gen_img / 2 + 0.5).clamp(0,1)
                        ct_vis   = (ct[j] / 2 + 0.5).clamp(0,1)

                        save_path = os.path.join(predict_dir, f"epoch_{epoch}_batch_{i}_img_{j}_guidance_{guidance_scale}.png")
                        torchvision.utils.save_image([cbct_vis, gen_vis, ct_vis], save_path, nrow=3)
                        saved_count += 1
                        if saved_count >= num_images_to_save:
                            break
                    if saved_count >= num_images_to_save:
                        break
            print(f"Saved {saved_count} images for epoch {epoch+1} to {predict_dir}")

    print("Training finished.")

def segmentation_losses(x_recon, ct_img, liver, tumor, ssim_map):    
    x_recon_liver_masked = x_recon * liver
    ct_img_liver_masked = ct_img * liver
    liver_mse_loss = F.mse_loss(x_recon_liver_masked, ct_img_liver_masked)
    liver_ssim_loss = 1 - torch.mean(ssim_map * liver)

    x_recon_tumor_masked = x_recon * tumor
    ct_img_tumor_masked = ct_img * tumor
    tumor_mse_loss = F.mse_loss(x_recon_tumor_masked, ct_img_tumor_masked)
    tumor_ssim_loss = 1 - torch.mean(ssim_map * tumor)
    
    return liver_mse_loss, tumor_mse_loss, liver_ssim_loss, tumor_ssim_loss


def train_segmentation_control(
    vae, 
    unet, 
    controlnet_cbct, 
    dr_module_cbct,
    controlnet_seg,
    dr_module_seg, 
    train_loader, 
    val_loader,
    test_loader, 
    epochs=1000, 
    save_dir='.', 
    predict_dir="predictions", 
    early_stopping=None, 
    patience=None, 
    gamma=1.0, 
    guidance_scale=1.0, 
    epochs_between_prediction=50, 
    learning_rate=5.0e-5, 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)
    amp_enabled = torch.cuda.is_available()
    vae.to(device)
    unet.to(device)
    controlnet_cbct.to(device)
    dr_module_cbct.to(device)
    controlnet_seg.to(device)
    dr_module_seg.to(device)

    # Collect trainable parameters per model
    controlnet_params = [p for p in controlnet_seg.parameters() if p.requires_grad]
    dr_module_params = [p for p in dr_module_seg.parameters() if p.requires_grad]
    params_to_train = controlnet_params + dr_module_params

    # Count them
    controlnet_param_count = sum(p.numel() for p in controlnet_params)
    dr_module_param_count = sum(p.numel() for p in dr_module_params)
    total_param_count = controlnet_param_count + dr_module_param_count

    # Print summary
    print(f"Trainable parameters in ControlNet: {controlnet_param_count}")
    print(f"Trainable parameters in DR Module:  {dr_module_param_count}")
    print(f"Total parameters to train:          {total_param_count}")
    
    optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate) # Use AdamW
    if patience is None:
        patience = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience,
        threshold=1e-4,
        min_lr=min(1e-7, learning_rate)
    )
    diffusion = Diffusion(device, timesteps=1000)

    # --- Training Loop ---
    best_val_loss = float('inf')
    early_stopping_counter = 0

    optimizer.zero_grad()

    for epoch in range(epochs):
        controlnet_seg.train()
        dr_module_seg.train()

        train_liver_mse_loss = 0
        train_tumor_mse_loss = 0
        train_liver_ssim_loss = 0
        train_tumor_ssim_loss = 0
        train_loss_total = 0

        # Monitoring losses
        train_diff_loss = 0
        train_global_ssim_loss = 0
        train_global_mse_loss = 0

        for i, (ct, cbct, segmentation, liver, tumor) in enumerate(train_loader):
            cbct_img = cbct.to(device)
            ct_img = ct.to(device)
            segmentation = segmentation.to(device)
            liver = liver.to(device)
            tumor = tumor.to(device)

            optimizer.zero_grad()
            
            with torch.no_grad():
                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

            # Forward pass
            t = diffusion.sample_timesteps(z_ct.size(0))
            noise = torch.randn_like(z_ct)
            z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

            controlnet_input_seg, _ = dr_module_seg(segmentation)
            down_res_samples_seg, middle_res_sample_seg = controlnet_seg(z_noisy_ct, controlnet_input_seg, t)
            
            with torch.no_grad():
                controlnet_input_cbct, _ = dr_module_cbct(cbct_img)
                down_res_samples_cbct, middle_res_sample_cbct = controlnet_cbct(z_noisy_ct, controlnet_input_cbct, t)
            
            pred_noise = unet(
                z_noisy_ct, 
                t, 
                down_res_samples_cbct, 
                middle_res_sample_cbct,
                down_res_samples_seg,
                middle_res_sample_seg
            )

            # Reconstruct image for segmentation losses
            alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1,1,1,1)
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
            z_hat = (z_noisy_ct - sqrt_one_minus_alpha_cumprod * pred_noise) / (sqrt_alpha_cumprod + 1e-8) # Added epsilon for stability
            x_recon = vae.decode(z_hat)

            # Compute losses
            g_ssim, ssim_map = ssim(ct_img, x_recon, data_range=2.0, full=True)
            liver_mse_loss, tumor_mse_loss, liver_ssim_loss, tumor_ssim_loss = segmentation_losses(x_recon, ct_img, liver, tumor, ssim_map)
            total_loss = liver_mse_loss + liver_ssim_loss + tumor_mse_loss + tumor_ssim_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()

            train_liver_mse_loss += liver_mse_loss.item()
            train_tumor_mse_loss += tumor_mse_loss.item()
            train_liver_ssim_loss += liver_ssim_loss.item()
            train_tumor_ssim_loss += tumor_ssim_loss.item()
            train_loss_total += total_loss.item()

            # Compute monitoring losses
            diff_loss = noise_loss(pred_noise, noise)
            global_ssim_loss = 1 - g_ssim
            global_mse_loss = F.mse_loss(x_recon, ct_img)

            train_diff_loss += diff_loss.item()
            train_global_ssim_loss += global_ssim_loss.item()
            train_global_mse_loss += global_mse_loss.item()
        
        # Average losses
        avg_train_loss_total = train_loss_total / len(train_loader)
        avg_train_liver_mse_loss = train_liver_mse_loss / len(train_loader)
        avg_train_tumor_mse_loss = train_tumor_mse_loss / len(train_loader)
        avg_train_liver_ssim_loss = train_liver_ssim_loss / len(train_loader)
        avg_train_tumor_ssim_loss = train_tumor_ssim_loss / len(train_loader)

        # Average monitoring losses
        avg_train_diff_loss = train_diff_loss / len(train_loader)
        avg_train_global_ssim_loss = train_global_ssim_loss / len(train_loader)
        avg_train_global_mse_loss = train_global_mse_loss / len(train_loader)

        # --- Validation Loop ---
        controlnet_seg.eval()
        dr_module_seg.eval()
        
        val_liver_mse_loss = 0
        val_tumor_mse_loss = 0
        val_liver_ssim_loss = 0
        val_tumor_ssim_loss = 0
        val_loss_total = 0

        val_diff_loss = 0
        val_global_ssim_loss = 0
        val_global_mse_loss = 0
        
        with torch.no_grad():
            for i, (ct, cbct, segmentation, liver, tumor) in enumerate(val_loader):
                cbct_img = cbct.to(device)
                ct_img = ct.to(device)
                segmentation = segmentation.to(device)
                liver = liver.to(device)
                tumor = tumor.to(device)

                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

                t = diffusion.sample_timesteps(z_ct.size(0))
                noise = torch.randn_like(z_ct)
                z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

                controlnet_input_seg, _ = dr_module_seg(segmentation)
                down_res_samples_seg, middle_res_sample_seg = controlnet_seg(z_noisy_ct, controlnet_input_seg, t)
                
                controlnet_input_cbct, _ = dr_module_cbct(cbct_img)
                down_res_samples_cbct, middle_res_sample_cbct = controlnet_cbct(z_noisy_ct, controlnet_input_cbct, t)
            
                pred_noise = unet(
                    z_noisy_ct, 
                    t, 
                    down_res_samples_cbct, 
                    middle_res_sample_cbct,
                    down_res_samples_seg,
                    middle_res_sample_seg
                )

                alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1,1,1,1)
                sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
                z_hat = (z_noisy_ct - sqrt_one_minus_alpha_cumprod * pred_noise) / (sqrt_alpha_cumprod + 1e-8)
                x_recon = vae.decode(z_hat)

                g_ssim, ssim_map = ssim(ct_img, x_recon, data_range=2.0, full=True)
                liver_mse_loss, tumor_mse_loss, liver_ssim_loss, tumor_ssim_loss = segmentation_losses(x_recon, ct_img, liver, tumor, ssim_map)
                total_loss = liver_mse_loss + liver_ssim_loss + tumor_mse_loss + tumor_ssim_loss

                val_liver_mse_loss += liver_mse_loss.item()
                val_tumor_mse_loss += tumor_mse_loss.item()
                val_liver_ssim_loss += liver_ssim_loss.item()
                val_tumor_ssim_loss += tumor_ssim_loss.item()
                val_loss_total += total_loss.item()

                diff_loss = noise_loss(pred_noise, noise)
                global_ssim_loss = 1 - g_ssim
                global_mse_loss = F.mse_loss(x_recon, ct_img)

                val_diff_loss += diff_loss.item()
                val_global_ssim_loss += global_ssim_loss.item()
                val_global_mse_loss += global_mse_loss.item()

        avg_val_loss_total = val_loss_total / len(val_loader)
        avg_val_liver_mse_loss = val_liver_mse_loss / len(val_loader)
        avg_val_tumor_mse_loss = val_tumor_mse_loss / len(val_loader)
        avg_val_liver_ssim_loss = val_liver_ssim_loss / len(val_loader)
        avg_val_tumor_ssim_loss = val_tumor_ssim_loss / len(val_loader)

        avg_val_diff_loss = val_diff_loss / len(val_loader)
        avg_val_global_ssim_loss = val_global_ssim_loss / len(val_loader)
        avg_val_global_mse_loss = val_global_mse_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | "
              f"Train Total Loss: {avg_train_loss_total:.6f}, Train Liver MSE: {avg_train_liver_mse_loss:.6f}, Train Tumor MSE: {avg_train_tumor_mse_loss:.6f}, Train Liver SSIM: {avg_train_liver_ssim_loss:.6f}, Train Tumor SSIM: {avg_train_tumor_ssim_loss:.6f} | "
              f"Train Diff Loss: {avg_train_diff_loss:.6f}, Train Global SSIM: {avg_train_global_ssim_loss:.6f}, Train Global MSE: {avg_train_global_mse_loss:.6f} | "
              f"Val Total Loss: {avg_val_loss_total:.6f}, Val Liver MSE: {avg_val_liver_mse_loss:.6f}, Val Tumor MSE: {avg_val_tumor_mse_loss:.6f}, Val Liver SSIM: {avg_val_liver_ssim_loss:.6f}, Val Tumor SSIM: {avg_val_tumor_ssim_loss:.6f} | "
              f"Val Diff Loss: {avg_val_diff_loss:.6f}, Val Global SSIM: {avg_val_global_ssim_loss:.6f}, Val Global MSE: {avg_val_global_mse_loss:.6f} | "
              f"LR: {current_lr:.1e}")
            
        scheduler.step(avg_val_loss_total)
        early_stopping_counter += 1
        
        if avg_val_loss_total < best_val_loss:
            best_val_loss = avg_val_loss_total
            early_stopping_counter = 0
            torch.save(controlnet_seg.state_dict(), os.path.join(save_dir, "controlnet_seg.pth"))
            torch.save(dr_module_seg.state_dict(), os.path.join(save_dir, "dr_module_seg.pth"))
            print(f"✅ Saved new best models at epoch {epoch+1} with val loss {avg_val_loss_total:.6f}")

        if early_stopping and early_stopping_counter >= early_stopping:
            print(f"Early stopped after {early_stopping} epochs with no improvement.")
            break

        # --- Inference/Saving Test Images ---
        if ((epoch + 1) % epochs_between_prediction == 0): # Save every 10 epochs
            print(f"--- Saving prediction for epoch {epoch+1} ---")

            controlnet_seg.eval()
            dr_module_seg.eval()

            num_images_to_save = 5
            saved_count = 0

            with torch.no_grad():
                for i, (ct, cbct, segmentation, _, _) in enumerate(test_loader):
                    ct = ct.to(device)
                    cbct = cbct.to(device)
                    segmentation.to(device)

                    controlnet_input_cbct, _ = dr_module_cbct(cbct)
                    controlnet_input_seg, _ = dr_module_seg(segmentation)

                    z_t = torch.randn_like(vae.encode(ct)[0])
                    T = diffusion.timesteps

                    for t_int in range(T - 1, -1, -1): 
                        t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                        # CFG: Predict noise twice
                        down_res_samples_cbct, middle_res_sample_cbct = controlnet_cbct(z_t, controlnet_input_cbct, t)
                        down_res_samples_seg, middle_res_sample_seg = controlnet_seg(z_t, controlnet_input_cbct, t)
                        pred_noise = unet(z_t, t, down_res_samples_cbct, middle_res_sample_cbct, down_res_samples_seg, middle_res_sample_seg)

                        # DDPM
                        beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                        alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                        alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                        sqrt_reciprocal_alpha_t = torch.sqrt(1.0 / alpha_t)

                        model_mean_coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
                        model_mean = sqrt_reciprocal_alpha_t * (z_t - model_mean_coef2 * pred_noise)

                        if t_int > 0:
                            variance = diffusion.beta[t_int].view(-1, 1, 1, 1) # Use posterior variance beta_t
                            noise = torch.randn_like(z_t)
                            z_t_minus_1 = model_mean + torch.sqrt(variance) * noise
                        else:
                            z_t_minus_1 = model_mean
                        z_t = z_t_minus_1

                    # Decode final latent
                    z_0 = z_t
                    generated_image_batch = vae.decode(z_0)

                    for j in range(generated_image_batch.size(0)):
                        generated_image = generated_image_batch[j]
                        cbct_image = cbct[j]
                        ct_image = ct[j]

                        generated_image_vis = (generated_image / 2 + 0.5).clamp(0, 1)
                        cbct_image_vis = (cbct_image / 2 + 0.5).clamp(0, 1)
                        ct_image_vis = (ct_image / 2 + 0.5).clamp(0, 1)

                        images_to_save = [cbct_image_vis, generated_image_vis, ct_image_vis]
                        save_filename = f"{predict_dir}/epoch_{epoch}_batch_{i}_img_{j}_guidance_scale_{guidance_scale}.png"

                        torchvision.utils.save_image(
                            images_to_save,
                            save_filename,
                            nrow=len(images_to_save),
                        )
                        saved_count += 1
                        if saved_count >= num_images_to_save:
                            break
                    if saved_count >= num_images_to_save:
                        break
            print(f"Saved {num_images_to_save} images for epoch {epoch+1} to {predict_dir}")
    print("Training finished.")

def test_dr_control_paca(
    vae, 
    unet, 
    controlnet, 
    dr_module, 
    test_loader, 
    predict_dir="predictions", 
    guidance_scales=[1.0],
    num_images_to_save=5
):
    os.makedirs(predict_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(device)
    unet.eval()
    controlnet.eval()
    dr_module.eval()
    vae.eval()

    saved_count = 0

    with torch.no_grad():
        for i, (ct, cbct) in enumerate(test_loader):
            ct = ct.to(device)
            cbct = cbct.to(device)

            controlnet_input, _ = dr_module(cbct)

            z_t = torch.randn_like(vae.encode(ct)[0])
            T = diffusion.timesteps

            for guidance_scale in guidance_scales:
                for t_int in tqdm(range(T - 1, -1, -1), total=T, desc=f"Sampling image", leave=False):
                    t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                    # CFG: Predict noise twice
                    down_res_samples, middle_res_sample = controlnet(z_t, controlnet_input, t)
                    pred_noise_cond = unet(z_t, t, down_res_samples, middle_res_sample)
                    pred_noise_uncond = unet(z_t, t, None, None)
                    pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

                    # DDPM
                    beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                    alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                    alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                    sqrt_reciprocal_alpha_t = torch.sqrt(1.0 / alpha_t)

                    model_mean_coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
                    model_mean = sqrt_reciprocal_alpha_t * (z_t - model_mean_coef2 * pred_noise)

                    if t_int > 0:
                        variance = diffusion.beta[t_int].view(-1, 1, 1, 1) # Use posterior variance beta_t
                        noise = torch.randn_like(z_t)
                        z_t_minus_1 = model_mean + torch.sqrt(variance) * noise
                    else:
                        z_t_minus_1 = model_mean
                    z_t = z_t_minus_1

                # Decode final latent
                z_0 = z_t
                generated_image_batch = vae.decode(z_0)

                for j in range(generated_image_batch.size(0)):
                    generated_image = generated_image_batch[j]
                    cbct_image = cbct[j]
                    ct_image = ct[j]

                    generated_image_vis = (generated_image / 2 + 0.5).clamp(0, 1)
                    cbct_image_vis = (cbct_image / 2 + 0.5).clamp(0, 1)
                    ct_image_vis = (ct_image / 2 + 0.5).clamp(0, 1)

                    images_to_save = [cbct_image_vis, generated_image_vis, ct_image_vis]
                    save_filename = f"{predict_dir}/batch_{i}_img_{j}_guidance_scale_{guidance_scale}.png"

                    torchvision.utils.save_image(
                        images_to_save,
                        save_filename,
                        nrow=len(images_to_save),
                    )
                    saved_count += 1
                    if saved_count >= num_images_to_save:
                        break
                if saved_count >= num_images_to_save:
                    break
            if saved_count >= num_images_to_save:
                break
    