import torch
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
from models.vae import VAE
from models.unet import UNet
from models.diffusion import Diffusion
from utils.dataset import PreprocessedCBCTtoCTDataset

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest_path = '../training_data/dataset_manifest.csv'
vae_weights_path = '../pretrained_models/vae.pth'
unet_weights_path = '../pretrained_models/unet.pth'
save_dir = 'unet_results'
pred_dir = f'{save_dir}/predictions'

subset_size = 5000
batch_size = 4
test_batch_size = 1
learning_rate = 5.0e-5
epochs = 200
num_workers = 8
max_grad_norm = 1.0
test_image_count = 3
gamma = 1

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

unet = UNet().to(device)
unet.load_state_dict(torch.load(unet_weights_path, map_location=device), strict=False) # Added key(s): "context_proj.weight", "context_proj.bias". 

diffusion = Diffusion(device, timesteps=1000)

params_to_train = list(unet.parameters())
print(f"Total parameters to train: {sum(p.numel() for p in params_to_train)}")

if not params_to_train:
     print("Error: No parameters selected for training. Check freezing logic.")
     exit()

optimizer = torch.optim.AdamW(params_to_train, lr=learning_rate) # Use AdamW
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=30,
    threshold=1e-4,
    verbose=True,
    min_lr=1e-7
)

# --- Training Loop ---
best_val_loss = float('inf')

for epoch in range(epochs):
    unet.train()
    train_loss = 0

    for cbct_img, ct_img in train_loader:
        optimizer.zero_grad()

        cbct_img = cbct_img.to(device)
        ct_img = ct_img.to(device)

        with torch.no_grad():
            ct_mu, ct_logvar = vae.encode(ct_img)
            z_ct = vae.reparameterize(ct_mu, ct_logvar)
            cbct_mu, cbct_logvar = vae.encode(cbct_img)

        t = diffusion.sample_timesteps(z_ct.size(0))
        noise = torch.randn_like(z_ct)
        z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

        pred_noise = unet(z_noisy_ct, cbct_mu, t)

        loss = noise_loss(pred_noise, noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=max_grad_norm)
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # --- Validation Loop ---
    unet.eval()
    val_loss = 0

    with torch.no_grad():
        for cbct_img, ct_img in val_loader:
            cbct_img = cbct_img.to(device)
            ct_img = ct_img.to(device)

            ct_mu, ct_logvar = vae.encode(ct_img)
            z_ct = vae.reparameterize(ct_mu, ct_logvar)
            cbct_mu, cbct_logvar = vae.encode(cbct_img)

            t = diffusion.sample_timesteps(z_ct.size(0))
            noise = torch.randn_like(z_ct)
            z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)

            pred_noise = unet(z_noisy_ct, cbct_mu, t)

            loss = noise_loss(pred_noise, noise)

            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.1e}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(unet.state_dict(), os.path.join(save_dir, "unet_cond.pth"))
        print(f"✅ Saved new best Unet Conditional model at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
        
    # --- Inference/Saving Test Images ---
    if ((epoch + 1) % 10 == 0): # Save every 10 epochs
        print(f"--- Saving prediction for epoch {epoch+1} ---")

        unet.eval()
        vae.eval()

        with torch.no_grad():
            for i, (cbct, ct) in enumerate(test_loader):
                if i >= test_image_count:
                    break

                cbct = cbct.to(device)
                ct = ct.to(device)

                cbct_mu, _ = vae.encode(cbct)

                z_t = torch.randn_like(vae.encode(ct)[0])
                T = diffusion.timesteps

                for t_int in range(T - 1, -1, -1): 
                    t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                    beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                    alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                    alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                    sqrt_reciprocal_alpha_t = torch.sqrt(1.0 / alpha_t)
                    pred_noise = unet(z_t, cbct_mu, t)
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




