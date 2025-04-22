import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models.diffusion import Diffusion
from quick_loop.blocks import nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, ControlNetPACAUpBlock
from quick_loop.degradationRemoval import degradation_loss
from quick_loop.unet import noise_loss

class UNetControlPaca(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=124,
                 dropout_rate=0.0):
        super().__init__()
        time_emb_dim = base_channels * 4

        ch1 = base_channels * 1
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4
        ch_mid = base_channels * 4

        attn_res_64 = False
        attn_res_32 = True
        attn_res_16 = True
        attn_res_8 = True

        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(ch1, ch2, time_emb_dim, attn_res_64, dropout_rate)
        self.down2 = DownBlock(ch2, ch3, time_emb_dim, attn_res_32, dropout_rate)
        self.down3 = DownBlock(ch3, ch4, time_emb_dim, attn_res_16, dropout_rate)
        self.down4 = DownBlock(ch4, ch_mid, time_emb_dim, attn_res_8, dropout_rate, downsample=False)

        self.middle = MiddleBlock(ch_mid, time_emb_dim, dropout_rate)

        self.up4 = ControlNetPACAUpBlock(ch_mid, ch4, ch_mid, time_emb_dim, attn_res_8, dropout_rate)
        self.up3 = ControlNetPACAUpBlock(ch4, ch3, ch4, time_emb_dim, attn_res_16, dropout_rate)
        self.up2 = ControlNetPACAUpBlock(ch3, ch2, ch3, time_emb_dim, attn_res_32, dropout_rate)
        self.up1 = ControlNetPACAUpBlock(ch2, ch1, ch2, time_emb_dim, attn_res_64, dropout_rate, upsample=False)

        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, down_additional_residuals, middle_additional_residual):
        additional_down_res_1_1, additional_down_res_1_2, additional_down_res_2_1, additional_down_res_2_2, additional_down_res_3_1, additional_down_res_3_2, additional_down_res_4_1, additional_down_res_4_2 = down_additional_residuals

        t_emb = self.time_embedding(t)
        h = self.init_conv(x)         

        h, (down_res_1_1, down_res_1_2) = self.down1(h, t_emb)
        h, (down_res_2_1, down_res_2_2) = self.down2(h, t_emb)
        h, (down_res_3_1, down_res_3_2) = self.down3(h, t_emb)
        h, (down_res_4_1, down_res_4_2) = self.down4(h, t_emb)

        h = self.middle(h, t_emb)
        h = h + middle_additional_residual

        skip_4_1 = torch.cat([down_res_4_1, additional_down_res_4_1], dim=1)
        skip_4_2 = torch.cat([down_res_4_2, additional_down_res_4_2], dim=1)
        h = self.up4(h, [skip_4_1, skip_4_2], [additional_down_res_4_1, additional_down_res_4_2], t_emb)

        skip_3_1 = torch.cat([down_res_3_1, additional_down_res_3_1], dim=1)
        skip_3_2 = torch.cat([down_res_3_2, additional_down_res_3_2], dim=1)
        h = self.up3(h, [skip_3_1, skip_3_2], [additional_down_res_3_1, additional_down_res_3_2], t_emb)

        skip_2_1 = torch.cat([down_res_2_1, additional_down_res_2_1], dim=1)
        skip_2_2 = torch.cat([down_res_2_2, additional_down_res_2_2], dim=1)
        h = self.up2(h, [skip_2_1, skip_2_2], [additional_down_res_2_1, additional_down_res_2_2], t_emb)

        skip_1_1 = torch.cat([down_res_1_1, additional_down_res_1_1], dim=1)
        skip_1_2 = torch.cat([down_res_1_2, additional_down_res_1_2], dim=1)
        h = self.up1(h, [skip_1_1, skip_1_2], [additional_down_res_1_1, additional_down_res_1_2], t_emb)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h
    
def load_unet_control_paca(unet_save_path=None, paca_save_path=None, unet_trainable=False, paca_trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unetControlPACA = UNetControlPaca().to(device)

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

    if unet_save_path:
        if unetControlPACA.parameters() - paca_params != unet_state_dict.parameters():
            print(f"WARNING: UNetControlPACA parameters - PACA parameters should be equal to the loaded state_dict parameters.")
            print(f"Loaded state_dict parameters: {unet_state_dict.parameters()}")
            print(f"UNetControlPACA parameters: {unetControlPACA.parameters()}")
            print(f"UNetControlPACA parameters - PACA parameters: {unetControlPACA.parameters() - paca_params}")

    print(f"Total UNetControlPACA parameters: {unetControlPACA.parameters()}")
    print(f"Unet parameters: {unetControlPACA.parameters() - paca_params}, trainable: {unet_trainable}")
    print(f"PACA parameters: {paca_params} , trainable: {paca_trainable}")

    return unetControlPACA

def train_dr_control_paca(vae, unet, controlnet, dr_module, train_loader, val_loader, epochs=1000, save_dir='.', predict_dir="predictions", early_stopping=None, patience=None, gamma=1.0, guidance_scale = 1.0):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)
    diffusion = Diffusion(device, timesteps=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not patience:
        patience = epochs
    params_to_train = list(controlnet.parameters()) + \
                  list(dr_module.parameters()) + \
                  [p for name, p in unet.named_parameters() if p.requires_grad]
    print(f"Total parameters to train: {sum(p.numel() for p in params_to_train)}")

    optimizer = torch.optim.AdamW(params_to_train, lr=5.0e-5) # Use AdamW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience,
        threshold=1e-4,
        verbose=True,
        min_lr=1e-7
    )

    # --- Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(epochs):
        unet.train()
        controlnet.train()
        dr_module.train()
        train_loss_total = 0
        train_loss_diff = 0
        train_loss_dr = 0

        for ct_img, cbct_img in train_loader:
            optimizer.zero_grad()

            cbct_img = cbct_img.to(device)
            ct_img = ct_img.to(device)

            with torch.no_grad():
                ct_mu, ct_logvar = vae.encode(ct_img)
                z_ct = vae.reparameterize(ct_mu, ct_logvar)

            controlnet_input, intermediate_preds = dr_module(cbct_img)
            loss_dr = degradation_loss(intermediate_preds, ct_img)

            t = diffusion.sample_timesteps(z_ct.size(0))
            noise = torch.randn_like(z_ct)
            z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

            control_features = controlnet(controlnet_input)
            pred_noise = unet(z_noisy_ct, t, control_features)

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

                control_features = controlnet(controlnet_input)

                pred_noise = unet(z_noisy_ct, t, control_features)

                loss_diff = noise_loss(pred_noise, noise)
                total_loss = loss_diff + gamma * loss_dr

                val_loss_total += total_loss.item()
                val_loss_diff += loss_diff.item()
                val_loss_dr += loss_dr.item()
        avg_val_loss_total = val_loss_total / len(val_loader)
        avg_val_loss_diff = val_loss_diff / len(val_loader)
        avg_val_loss_dr = val_loss_dr / len(val_loader)
        
        scheduler.step(avg_val_loss_total)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss_total:.4f} (Diff: {avg_train_loss_diff:.4f}, DR: {avg_train_loss_dr:.4f}) | "
            f"Val Loss: {avg_val_loss_total:.4f} (Diff: {avg_val_loss_diff:.4f}, DR: {avg_val_loss_dr:.4f}) | LR: {current_lr:.1e}")
        
        if avg_val_loss_total < best_val_loss:
            best_val_loss = avg_val_loss_total
            
            torch.save(controlnet.state_dict(), f"{save_dir}/controlnet.pth")
            torch.save(dr_module.state_dict(), f"{save_dir}/dr_module.pth")
            # Save only PACA layer parameters from UNet
            paca_state_dict = {k: v for k, v in unet.state_dict().items() if 'paca' in k.lower()}
            if paca_state_dict: # Save only if PACA layers exist
                torch.save(paca_state_dict, f"{save_dir}/paca_layers.pth")
                print(f"✅ Saved new best ControlNet+PACA model at epoch {epoch+1} with val loss {avg_val_loss_total:.4f}")
            else:
                print(f"✅ Saved new best ControlNet+DR model (no PACA found/saved) at epoch {epoch+1} with val loss {avg_val_loss_total:.4f}")

        # --- Inference/Saving Test Images ---
        if ((epoch + 1) % 10 == 0): # Save every 10 epochs
            print(f"--- Saving prediction for epoch {epoch+1} ---")

            unet.eval()
            controlnet.eval()
            dr_module.eval()
            vae.eval()

            with torch.no_grad():
                for i, (cbct, ct) in enumerate(val_loader):
                    cbct = cbct.to(device)
                    ct = ct.to(device)

                    controlnet_input, _ = dr_module(cbct)
                    control_features = controlnet(controlnet_input)

                    z_t = torch.randn_like(vae.encode(ct)[0])
                    T = diffusion.timesteps

                    for t_int in range(T - 1, -1, -1): 
                        t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                        beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                        alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                        alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                        sqrt_reciprocal_alpha_t = torch.sqrt(1.0 / alpha_t)
                        pred_noise = unet(z_t, t, control_features)
                        model_mean_coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
                        model_mean = sqrt_reciprocal_alpha_t * (z_t - model_mean_coef2 * pred_noise)

                        if t_int > 0:
                            variance = diffusion.beta[t_int].view(-1, 1, 1, 1)
                            noise = torch.randn_like(z_t)
                            z_t_minus_1 = model_mean + torch.sqrt(variance) * noise
                        else:
                            z_t_minus_1 = model_mean
                        z_t = z_t_minus_1

                    z_0 = z_t
                    generated_image = vae.decode(z_0)

                    generated_image_vis = (generated_image / 2 + 0.5).clamp(0, 1).squeeze(0)
                    cbct_image_vis = (cbct / 2 + 0.5).clamp(0, 1).squeeze(0)
                    ct_image_vis = (ct / 2 + 0.5).clamp(0, 1).squeeze(0)

                    images_to_save = [cbct_image_vis, generated_image_vis, ct_image_vis]
                    save_filename = f"{predict_dir}/epoch_{epoch+1}_img_{i+1}_comparison.png"
                    torchvision.utils.save_image(
                        images_to_save,
                        save_filename,
                        nrow=len(images_to_save),
                    )
                    print(f"Saved comparison image to {save_filename}")

    print("Training finished.")
    
