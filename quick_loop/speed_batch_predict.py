import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
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
VOLUME_INDICES = [3]
OUT_DIR = '../predictions490/'

GUIDANCE_SCALE = 1.0
BATCH_SIZE = 32  # tune as needed
# DDIM / schedule parameters: reduce steps for faster inference
DDIM_STEPS = 40     # total coarse sampling steps
POWER_P = 2.0       # power-law exponent for smoothing
FINE_CUTOFF = 9     # switch to single-step updates at t<=9 (last 10 steps)

MODELS_PATH = 'controlnet_training/v2/'
VAE_SAVE_PATH = os.path.join(MODELS_PATH, 'vae.pth')
UNET_SAVE_PATH = os.path.join(MODELS_PATH, 'unet.pth')
PACA_LAYERS_SAVE_PATH = os.path.join(MODELS_PATH, 'paca_layers.pth')
CONTROLNET_SAVE_PATH = os.path.join(MODELS_PATH, 'controlnet.pth')
DEGRADATION_REMOVAL_SAVE_PATH = os.path.join(MODELS_PATH, 'dr_module.pth')

# ------------------------
# Utility Functions
# ------------------------

def load_volume_slices(volume_dir: str, transform=None):
    """
    Load all .npy slices in volume_dir, apply transform and return list of (filename, tensor).
    """
    slice_files = sorted([f for f in os.listdir(volume_dir) if f.endswith('.npy')])
    slices = []
    for fname in slice_files:
        arr = np.load(os.path.join(volume_dir, fname)).astype(np.float32) / 1000.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if transform:
            tensor = transform(tensor)
        slices.append((fname, tensor))
    return slices


def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ------------------------
# Schedule Helper
# ------------------------

def make_mixed_schedule(T=1000, N=DDIM_STEPS, p=POWER_P, fine_cutoff=FINE_CUTOFF):
    """
    Build a DDIM timetable that uses a power-law taper down to `fine_cutoff`,
    then single-step from fine_cutoff->0.
    """
    idx = np.arange(N + 1)
    raw = (1 - (idx / N) ** p) * T
    smooth_ts = np.unique(raw.astype(int))[::-1]
    smooth_ts = smooth_ts[smooth_ts > fine_cutoff]
    if smooth_ts.size == 0 or smooth_ts[0] != T:
        smooth_ts = np.concatenate(([T], smooth_ts))
    fine_ts = np.arange(fine_cutoff, -1, -1)
    return np.concatenate((smooth_ts, fine_ts))

# ------------------------
# Inference Function
# ------------------------

def predict_volume(
    vae, unet, controlnet, dr_module,
    cbct_slices,
    save_dir: str,
    guidance_scale: float,
    batch_size: int,
    ddim_steps: int = DDIM_STEPS,
    power_p: float = POWER_P,
    fine_cutoff: int = FINE_CUTOFF
):
    """
    Run batched, half-precision DDIM inference on CBCT slices and save predictions,
    tracking time per batch and per volume.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    diffusion = Diffusion(device)
    betas = diffusion.beta.to(device).half()
    alpha_cumprod = diffusion.alpha_cumprod.to(device).half()
    T = diffusion.timesteps
    T_max = T - 1

    vae.to(device).eval()
    unet.to(device).eval()
    controlnet.to(device).eval()
    dr_module.to(device).eval()

    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        controlnet = nn.DataParallel(controlnet)
        dr_module = nn.DataParallel(dr_module)

    os.makedirs(save_dir, exist_ok=True)
    schedule = make_mixed_schedule(T=T_max, N=ddim_steps, p=power_p, fine_cutoff=fine_cutoff)

    # Start volume timer
    volume_start = time.time()

    for batch_idx, batch in enumerate(chunks(cbct_slices, batch_size), start=1):
        batch_start = time.time()
        names = [fn for fn, _ in batch]
        imgs = torch.stack([t for _, t in batch], dim=0).to(device).half()

        with torch.inference_mode(), torch.cuda.amp.autocast():
            control_inputs, _ = dr_module(imgs)
            mu, logvar = vae.encode(imgs)
            z = torch.randn_like(mu)

            for i in range(len(schedule) - 1):
                t, t_prev = int(schedule[i]), int(schedule[i + 1])
                t_tensor = torch.full((z.size(0),), t, device=device, dtype=torch.long)
                down_res, mid_res = controlnet(z, control_inputs, t_tensor)
                eps = unet(z, t_tensor, down_res, mid_res)

                a_t = alpha_cumprod[t]
                a_prev = alpha_cumprod[t_prev]
                sqrt_a_t = a_t.sqrt()
                sqrt_one_minus_a_t = (1 - a_t).sqrt()

                z = ((z - sqrt_one_minus_a_t * eps) / sqrt_a_t) * a_prev.sqrt() \
                    + (1 - a_prev).sqrt() * eps

            gen = vae.decode(z)

        # Save outputs
        out_np = (gen.cpu().float().numpy() * 1000.0)
        for i, fname in enumerate(names):
            np.save(os.path.join(save_dir, fname), out_np[i].squeeze(0))

        # Log batch time
        batch_time = time.time() - batch_start
        print(f"Volume {os.path.basename(save_dir)}, batch {batch_idx} took {batch_time:.2f}s")

    # Log volume time
    volume_time = time.time() - volume_start
    print(f"Volume {os.path.basename(save_dir)} processed in {volume_time:.2f}s")

# ------------------------
# Main
# ------------------------

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Pad((0, 64, 0, 64), fill=-1),
        transforms.Resize((256, 256)),
    ])

    vae = load_vae(VAE_SAVE_PATH).half()
    unet = load_unet_control_paca(UNET_SAVE_PATH, PACA_LAYERS_SAVE_PATH).half()
    controlnet = load_controlnet(CONTROLNET_SAVE_PATH).half()
    dr_module = load_degradation_removal(DEGRADATION_REMOVAL_SAVE_PATH).half()

    for vol in VOLUME_INDICES:
        cbct_folder = os.path.join(CBCT_DIR, f"volume-{vol}")
        save_folder = os.path.join(OUT_DIR, f"volume-{vol}")
        cbct_slices = load_volume_slices(cbct_folder, transform)
        predict_volume(
            vae, unet, controlnet, dr_module,
            cbct_slices, save_folder,
            GUIDANCE_SCALE, BATCH_SIZE
        )
    print("All volumes processed.")
