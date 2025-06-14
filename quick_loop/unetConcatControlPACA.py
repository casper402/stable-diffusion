import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion import Diffusion
from quick_loop.blocks import ControlNetPACAUpBlock, nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, ConditionalUpBlock, UpBlock, PACALayer
from quick_loop.degradationRemoval import degradation_loss
from quick_loop.unet import noise_loss
from utils.losses import PerceptualLoss, SsimLoss
from torch.optim import AdamW

class UNetConcatControlPACA(nn.Module):
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
        self.init_conv = nn.Conv2d(in_channels*2, ch1, kernel_size=3, padding=1)

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

    def forward(self, x, condition, t, down_paca_control_residuals=None, middle_paca_control_residual=None, down_control_residuals=None, middle_control_residual=None):
        control_paca = True if down_paca_control_residuals is not None else False
        extra_control = True if down_control_residuals is not None else False

        if control_paca:
            additional_down_res_1_1, additional_down_res_1_2, additional_down_res_2_1, additional_down_res_2_2, additional_down_res_3_1, additional_down_res_3_2, additional_down_res_4_1, additional_down_res_4_2 = down_paca_control_residuals

        if extra_control:
            extra_additional_down_res_1_1, extra_additional_down_res_1_2, extra_additional_down_res_2_1, extra_additional_down_res_2_2, extra_additional_down_res_3_1, extra_additional_down_res_3_2, extra_additional_down_res_4_1, extra_additional_down_res_4_2 = down_control_residuals
        
        t_emb = self.time_embedding(t)
        x = torch.cat((x, condition), dim=1)
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
            middle_paca_control_residual += middle_control_residual

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
    
def load_unet_concat_control_paca(unet_save_path=None, paca_save_path=None, unet_trainable=False, paca_trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unetConcatControlPACA = UNetConcatControlPACA().to(device)

    if paca_save_path:
        paca_state_dict = torch.load(paca_save_path, map_location=device)
        _, paca_unexpected_keys = unetConcatControlPACA.load_state_dict(paca_state_dict, strict=False)
        if paca_unexpected_keys:
            print(f"Unexpected keys in PACA state_dict: {paca_unexpected_keys}")

    if unet_save_path is None:
        print("UNet initialized with random weights.")
    else: 
        unet_state_dict = torch.load(unet_save_path, map_location=device)
        _, unetControlPACA_unexpected_keys = unetConcatControlPACA.load_state_dict(unet_state_dict, strict=False)
        if unetControlPACA_unexpected_keys:
            print(f"Unexpected keys in UNetControlPACA state_dict: {unetControlPACA_unexpected_keys}")

    for param in unetConcatControlPACA.parameters():
        param.requires_grad = unet_trainable
    
    paca_params = 0
    for name, param in unetConcatControlPACA.named_parameters():
        if 'paca' in name.lower():
            param.requires_grad = paca_trainable
            paca_params += param.numel()
    
    unet_control_paca_params = sum(p.numel() for p in unetConcatControlPACA.parameters())
    unet_params = sum(p.numel() for p in unet_state_dict.values())
    if unet_save_path:
        if unet_control_paca_params - paca_params != unet_params:
            print(f"WARNING: UNetControlPACA parameters - PACA parameters should be equal to the loaded state_dict parameters.")
            print(f"Loaded state_dict parameters: {unet_params}")
            print(f"UNetControlPACA parameters: {unet_control_paca_params}")
            print(f"UNetControlPACA parameters - PACA parameters: {unet_control_paca_params - paca_params}")
        print(f"UNetControlPACA loaded from {unet_save_path}")

    return unetConcatControlPACA

def train_unet_concat_control_paca(
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
    learning_rate=5e-6, 
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

                cbct_z_mu, cbct_z_logvar = vae.encode(cbct_img)
                cbct_z = vae.reparameterize(cbct_z_mu, cbct_z_logvar)

            # Forward pass
            controlnet_input, intermediate_preds = dr_module(cbct_img)
            t = diffusion.sample_timesteps(z_ct.size(0))
            noise = torch.randn_like(z_ct)
            z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)
            down_res_samples, middle_res_sample = controlnet(z_noisy_ct, controlnet_input, t)
            pred_noise = unet(z_noisy_ct, cbct_z, t, down_res_samples, middle_res_sample)

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

        val_generator = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            for ct_img, cbct_img in val_loader:
                cbct_img = cbct_img.to(device)
                ct_img = ct_img.to(device)

                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

                cbct_z_mu, cbct_z_logvar = vae.encode(cbct_img)
                cbct_z = vae.reparameterize(cbct_z_mu, cbct_z_logvar)

                controlnet_input, intermediate_preds = dr_module(cbct_img)

                loss_dr = degradation_loss(intermediate_preds, ct_img)

                t = diffusion.sample_timesteps(z_ct.size(0), generator=val_generator)
                noise = torch.randn(
                    z_ct.shape,
                    device=z_ct.device,
                    dtype=z_ct.dtype,
                    generator=val_generator
                )
                z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

                down_res_samples, middle_res_sample = controlnet(z_noisy_ct, controlnet_input, t)
                pred_noise = unet(z_noisy_ct, cbct_z, t, down_res_samples, middle_res_sample)

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

    print("Training finished.")