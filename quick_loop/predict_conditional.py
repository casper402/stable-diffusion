import os
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from models.diffusion import Diffusion
from quick_loop.vae import load_vae
from quick_loop.unetConditional import load_cond_unet

# ------------------------
# Configuration Variables
# ------------------------
CBCT_DIR = '../training_data/CBCT/test/'
# Possible volumes: 3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129
VOLUME_IDX = 8  # choose from the list above
OUT_DIR = 'conditional_unet_base_channels_256/inference/'

# Single guidance scale (set to 1.0)
GUIDANCE_SCALE = 1.0
# Number of slices to process in one batch (tune based on your GPU memory)
BATCH_SIZE = 1

VAE_SAVE_PATH = '../pretrained_models/vae.pth'
UNET_SAVE_PATH = 'conditional_unet_base_channels_256/unet.pth'

def load_volume_slices(volume_dir: str, transform=None):
    """
    Load all .npy slices in volume_dir, apply transform and return list of (filename, tensor).
    """
    slice_files = sorted([f for f in os.listdir(volume_dir) if f.endswith('.npy')])
    slices = []
    for fname in slice_files:
        arr = np.load(os.path.join(volume_dir, fname)).astype(np.float32) / 1000.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # add channel dim
        if transform:
            tensor = transform(tensor)
        slices.append((fname, tensor))
    return slices

def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def predict_volume(
    vae, 
    unet,
    cbct_slices,
    save_dir: str,
    guidance_scale: float,
    batch_size: int
):
    """
    Run batched, mixed-precision inference on CBCT slices and save predictions.
    """
    # Set up device and diffusion scheduler
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    diffusion = Diffusion(device)

    # Preload constants on GPU
    betas = diffusion.beta.to(device)
    alphas = diffusion.alpha.to(device)
    alpha_cumprod = diffusion.alpha_cumprod.to(device)
    timesteps = diffusion.timesteps

    # Move models to device (no DataParallel)
    vae.to(device).eval()
    unet.to(device).eval()

    os.makedirs(save_dir, exist_ok=True)

    total_batches = math.ceil(len(cbct_slices) / batch_size)

    for batch_idx, batch in enumerate(chunks(cbct_slices, batch_size), start=1):
        print(f"Processing batch {batch_idx}/{total_batches}…")

        names = [item[0] for item in batch]
        imgs = torch.stack([item[1] for item in batch], dim=0).to(device)  # (B,1,H,W)

        with torch.no_grad():
            mu, logvar = vae.encode(imgs)
            z_cond = vae.reparameterize(mu, logvar)
            z_t = torch.randn_like(mu)

            # 3) Reverse diffusion
            print(f"Batch {batch_idx}: z_cond stats: min={z_cond.min().item():.4f}, max={z_cond.max().item():.4f}, has_nan={torch.isnan(z_cond).any()}")
            for t_int in reversed(range(0, timesteps, 250)):
                print(f"  t={t_int}: z_t stats: min={z_t.min().item():.4f}, max={z_t.max().item():.4f}, has_nan={torch.isnan(z_t).any()}")
                t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)
                print(f"    Params: beta_t={beta_t.item():.6f}, alpha_t={alpha_t.item():.6f}, alpha_c={alpha_c.item():.6f}")

                # ControlNet guidance
                pred_cond = unet(z_t, z_cond, t)

                # With guidance scale 1.0, we only need the conditioned output
                pred_noise = pred_cond

                # Compute update
                beta_t = betas[t_int].view(-1, 1, 1, 1)
                alpha_t = alphas[t_int].view(-1, 1, 1, 1)
                alpha_c = alpha_cumprod[t_int].view(-1, 1, 1, 1)
                coef = beta_t / torch.sqrt(1.0 - alpha_c)
                mean = torch.sqrt(1.0 / alpha_t) * (z_t - coef * pred_noise)

                if t_int > 0:
                    noise = torch.randn_like(z_t)
                    z_t = mean + torch.sqrt(beta_t) * noise
                else:
                    z_t = mean
                print(f"  t={t_int} (end): z_t stats: min={z_t.min().item():.4f}, max={z_t.max().item():.4f}, has_nan={torch.isnan(z_t).any()}")

            # 4) VAE decode
            print(f"Batch {batch_idx}: Final z_t stats: min={z_t.min().item():.4f}, max={z_t.max().item():.4f}, has_nan={torch.isnan(z_t).any()}")
            gen = vae.decode(z_t)
            print(f"Batch {batch_idx}: Decoded gen stats: min={gen.min().item():.4f}, max={gen.max().item():.4f}, has_nan={torch.isnan(gen).any()}")

        # 5) Save outputs (scaled to [-1000, 1000])
        out_np = gen.cpu().numpy() * 1000.0
        for i, fname in enumerate(names):
            out = out_np[i].squeeze(0)
            save_path = os.path.join(save_dir, fname)
            np.save(save_path, out)

    print(f"Saved predictions to {save_dir}")

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Pad((0, 64, 0, 64), fill=-1),
        transforms.Resize((256, 256)),
    ])

    cbct_folder = os.path.join(CBCT_DIR, f"volume-{VOLUME_IDX}")
    save_folder = os.path.join(OUT_DIR, f"volume-{VOLUME_IDX}")

    cbct_slices = load_volume_slices(cbct_folder, transform)

    vae = load_vae(VAE_SAVE_PATH)
    unet = load_cond_unet(
        save_path=UNET_SAVE_PATH,
        base_channels=256,
    )

    predict_volume(
        vae, 
        unet,
        cbct_slices,
        save_folder,
        GUIDANCE_SCALE,
        BATCH_SIZE
    )
