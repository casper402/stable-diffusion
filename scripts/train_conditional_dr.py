import torch
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from models.vae import VAE
from models.conditional import UNetPACA, ControlNet, DegradationRemovalModuleResnet
from models.diffusion import Diffusion

from utils.dataset import PreprocessedCBCTtoCTDataset, CBCTtoCTDataset

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)

def degradation_loss(intermediate_preds, gt_ct_img):
    """Calculates the L1 loss for intermediate DR predictions."""
    pred_128, pred_64, pred_32 = intermediate_preds

    # Downsample ground truth CT image (ensure it's [B, 1, H, W])
    # Using 'area' interpolation for downsampling images is often good
    gt_128 = F.interpolate(gt_ct_img, size=(128, 128), mode='area')
    gt_64 = F.interpolate(gt_ct_img, size=(64, 64), mode='area')
    gt_32 = F.interpolate(gt_ct_img, size=(32, 32), mode='area')

    loss_128 = F.l1_loss(pred_128, gt_128)
    loss_64 = F.l1_loss(pred_64, gt_64)
    loss_32 = F.l1_loss(pred_32, gt_32)

    # Combine losses (equal weighting here, can be adjusted)
    total_dr_loss = loss_128 + loss_64 + loss_32
    return total_dr_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest_path = '../training_data/dataset_manifest.csv'
vae_weights_path = '../pretrained_models/vae.pth'
unet_weights_path = '../pretrained_models/unet.pth'
dr_weights_path = '../pretrained_models/dr_module.pth'
save_dir = 'dr_results_2'
pred_dir = f'{save_dir}/predictions'

subset_size = 5000
batch_size = 4
test_batch_size = 1
learning_rate = 5.0e-5
epochs = 200
num_workers = 8
max_grad_norm = 1.0
test_image_count = 3
gamma = 1 # Weight for the degradation loss (as used in PASD paper)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad((0, 64, 0, 64)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

# CBCT_path = '../training_data/CBCT'
# CT_path = '../training_data/CT'
# dataset = CBCTtoCTDataset(CBCT_path=CBCT_path, CT_path=CT_path, transform=transform)

dataset = PreprocessedCBCTtoCTDataset(manifest_path=manifest_path, transform=transform)
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

unet = UNetPACA().to(device)
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
    print("Warning: No PACA parameters found or unfrozen in UNet")

controlnet = ControlNet().to(device)
controlnet.load_state_dict(torch.load(unet_weights_path, map_location=device), strict=False)
controlnet.train()
trainable_controlnet_params = sum(p.numel() for p in controlnet.parameters())
print(f"ControlNet instantiated ({trainable_controlnet_params} trainable parameters).")

dr_module = DegradationRemovalModuleResnet(in_channels=1, final_out_channels=4).to(device) # Matches ControlNet input
dr_module.load_state_dict(torch.load(dr_weights_path, map_location=device), strict=False)
dr_module.train()
trainable_dr_params = sum(p.numel() for p in dr_module.parameters() if p.requires_grad)
print(f"DR Module instantiated ({trainable_dr_params} trainable parameters).")

diffusion = Diffusion(device, timesteps=1000)

params_to_train = list(controlnet.parameters()) + \
                  list(dr_module.parameters()) + \
                  [p for name, p in unet.named_parameters() if p.requires_grad]
print(f"Total parameters to train: {sum(p.numel() for p in params_to_train)}")

if not params_to_train:
     print("Error: No parameters selected for training. Check freezing logic.")
     exit()

optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate) # Use AdamW
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20,
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

    for cbct_img, ct_img in train_loader:
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
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=max_grad_norm)
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
        for cbct_img, ct_img in val_loader:
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
             torch.save(paca_state_dict, f"{save_dir}/unet_paca_layers.pth")
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
            for i, (cbct, ct) in enumerate(test_loader):
                if i >= test_image_count:
                    break

                cbct = cbct.to(device)
                ct = ct.to(device)

                controlnet_input, _ = dr_module(cbct)
                controlnet_features = controlnet(controlnet_input)

                z_t = torch.randn_like(vae.encode(ct)[0])
                T = diffusion.timesteps

                for t_int in range(T - 1, -1, -1): 
                    t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                    pred_noise = unet(z_t, t, controlnet_features)

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
                save_filename = f"{pred_dir}/epoch_{epoch+1}_img_{i+1}_comparison.png"
                torchvision.utils.save_image(
                    images_to_save,
                    save_filename,
                    nrow=len(images_to_save),
                )
                print(f"Saved comparison image to {save_filename}")

print("Training finished.")




