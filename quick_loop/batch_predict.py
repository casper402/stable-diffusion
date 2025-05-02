import os
import numpy as np
import torch
import torch.nn as nn
import math
from torchvision import transforms
from models.diffusion import Diffusion
from quick_loop.vae import load_vae
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from quick_loop.unetControlPACA import load_unet_control_paca

# ------------------------
# Configuration Variables
# ------------------------
CBCT_DIR = '../training_data/scaled-490/'
# Possible volumes to process
# VOLUME_INDICES = [3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129]
VOLUME_INDICES = [3, 8]
OUT_DIR = '../predictions490/'

# Single guidance scale (set to 1.0)
GUIDANCE_SCALE = 1.0
# Number of slices to process in one batch (tune based on your GPU memory)
BATCH_SIZE = 128

MODELS_PATH = 'controlnet_training/v2/'
VAE_SAVE_PATH = os.path.join(MODELS_PATH, 'vae.pth')
UNET_SAVE_PATH = os.path.join(MODELS_PATH, 'unet.pth')
PACA_LAYERS_SAVE_PATH = os.path.join(MODELS_PATH, 'paca_layers.pth')
CONTROLNET_SAVE_PATH = os.path.join(MODELS_PATH, 'controlnet.pth')
DEGRADATION_REMOVAL_SAVE_PATH = os.path.join(MODELS_PATH, 'dr_module.pth')

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
    vae, unet, controlnet, dr_module,
    cbct_slices,
    save_dir: str,
    guidance_scale: float,
    batch_size: int
):
    """
    Run batched, mixed-precision inference on CBCT slices and save predictions.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    diffusion = Diffusion(device)

    # Preload constants on GPU
    betas = diffusion.beta.to(device)
    alphas = diffusion.alpha.to(device)
    alpha_cumprod = diffusion.alpha_cumprod.to(device)
    timesteps = diffusion.timesteps

    # Move models to device and set eval
    vae.to(device).eval()
    unet.to(device).eval()
    controlnet.to(device).eval()
    dr_module.to(device).eval()

    # Wrap heavy models for multi-GPU if available
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        controlnet = nn.DataParallel(controlnet)
        dr_module = nn.DataParallel(dr_module)

    os.makedirs(save_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    total_batches = math.ceil(len(cbct_slices) / batch_size)

    for batch_idx, batch in enumerate(chunks(cbct_slices, batch_size), start=1):
        print(f"Processing batch {batch_idx}/{total_batches} for {save_dir}…")

        names = [item[0] for item in batch]
        imgs = torch.stack([item[1] for item in batch], dim=0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # 1) Degradation removal / control inputs
            control_inputs, _ = dr_module(imgs)

            # 2) VAE encode
            mu, logvar = vae.encode(imgs)
            z_t = torch.randn_like(mu)

            # 3) Reverse diffusion
            for t_int in reversed(range(timesteps)):
                t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                down_res, mid_res = controlnet(z_t, control_inputs, t)
                pred_cond = unet(z_t, t, down_res, mid_res)

                pred_noise = pred_cond

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

            # 4) VAE decode
            gen = vae.decode(z_t)

        # 5) Save outputs
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

    # Load models once
    vae = load_vae(VAE_SAVE_PATH)
    unet = load_unet_control_paca(
        unet_save_path=UNET_SAVE_PATH,
        paca_save_path=PACA_LAYERS_SAVE_PATH
    )
    controlnet = load_controlnet(save_path=CONTROLNET_SAVE_PATH)
    dr_module = load_degradation_removal(save_path=DEGRADATION_REMOVAL_SAVE_PATH)

    # Loop over all specified volumes
    for vol_idx in VOLUME_INDICES:
        print(f"\nStarting inference for volume {vol_idx}")
        cbct_folder = os.path.join(CBCT_DIR, f"volume-{vol_idx}")
        save_folder = os.path.join(OUT_DIR, f"volume-{vol_idx}")

        # Load slices and run prediction
        cbct_slices = load_volume_slices(cbct_folder, transform)
        predict_volume(
            vae, unet, controlnet, dr_module,
            cbct_slices,
            save_folder,
            GUIDANCE_SCALE,
            BATCH_SIZE
        )
    print("All volumes processed.")
