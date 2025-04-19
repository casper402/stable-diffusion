import os
import torch
import torchvision

from utils.dataset import get_dataloaders, CTDatasetNPY, PairedCTCBCTDatasetNPY
from models.diffusion import Diffusion
from quick_loop.vae import load_vae, train_vae
from quick_loop.unet import load_unet, train_unet, noise_loss
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal, degradation_loss
from quick_loop.unetControlPACA import load_unet_control_paca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "data_quick_loop"

vae_predict_dir = os.path.join(save_dir, "vae_predictions")
unet_predict_dir = os.path.join(save_dir, "unet_predictions")
conditional_predict_dir = os.path.join(save_dir, "conditional_predictions")

vae_save_path = os.path.join(save_dir, "vae.pth")
unet_save_path = os.path.join(save_dir, "unet.pth")
controlnet_save_path = os.path.join(save_dir, "controlnet.pth")
paca_layers_save_path = os.path.join(save_dir, "paca_layers.pth")
degradation_removal_save_path = os.path.join(save_dir, "degradation_removal.pth")

os.makedirs(save_dir, exist_ok=True)

manifest_path = "../data_quick_loop/manifest.csv"
train_loader, val_loader, _ = get_dataloaders(manifest_path, batch_size=2, num_workers=2, dataset_class=CTDatasetNPY)

vae = load_vae(trainable=True)
train_vae(vae=vae, train_loader=train_loader, val_loader=val_loader, epochs=1, early_stopping=50, save_path=os.path.join(save_dir, "vae.pth"), predict_dir=vae_predict_dir)

unet = load_unet(trainable=True)
train_unet(unet=unet, train_loader=train_loader, val_loader=val_loader, epochs=1, early_stopping=50, save_path=os.path.join(save_dir, "unet.pth"), predict_dir=unet_predict_dir)

vae = load_vae(save_path=vae_save_path, trainable=False)
controlnet = load_controlnet(trainable=True)
dr_module = load_degradation_removal(trainable=True)
unet = load_unet_control_paca(unet_save_path=unet_save_path, paca_trainable=True)

train_loader, val_loader, _ = get_dataloaders(manifest_path, batch_size=2, num_workers=2, dataset_class=PairedCTCBCTDatasetNPY)

def train_conditional(vae, unet, controlnet, dr_module, train_loader, val_loader, epochs=1000, save_dir='.', predict_dir="predictions", early_stopping=None, patience=None, gamma=1.0, guidance_scale = 1.0):
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

train_conditional(vae=vae, unet=unet, controlnet=controlnet, dr_module=dr_module, train_loader=train_loader, val_loader=val_loader, epochs=1, save_dir=save_dir, predict_dir=conditional_predict_dir, early_stopping=None, patience=None, gamma=1.0)
print("All trainings finished.")