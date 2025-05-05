import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
VOLUME_INDICES = [3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129]
OUT_DIR = '../predictionsV2-490/'

GUIDANCE_SCALE = 1.0
BATCH_SIZE = 32  # tune as needed
# DDIM / schedule parameters: reduce steps for faster inference
DDIM_STEPS = 40     # total coarse sampling steps
POWER_P = 2.0       # power-law exponent for smoothing
FINE_CUTOFF = 9     # switch to single-step updates at t<=9 (last 10 steps)

MODELS_PATH = 'controlnet_v2/'
VAE_SAVE_PATH = os.path.join(MODELS_PATH, 'vae_joint_vae.pth')
UNET_SAVE_PATH = os.path.join(MODELS_PATH, 'unet_joint_unet.pth')
PACA_LAYERS_SAVE_PATH = os.path.join(MODELS_PATH, 'paca_layers.pth')
CONTROLNET_SAVE_PATH = os.path.join(MODELS_PATH, 'controlnet.pth')
DEGRADATION_REMOVAL_SAVE_PATH = os.path.join(MODELS_PATH, 'dr_module.pth')

# ------------------------
# Dataset for CBCT slices
# ------------------------
class CBCTDatasetNPY(Dataset):
    def __init__(self, volume_dir: str, transform=None):
        self.files = sorted([f for f in os.listdir(volume_dir) if f.endswith('.npy')])
        self.volume_dir = volume_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        arr = np.load(os.path.join(self.volume_dir, fname)).astype(np.float32) / 1000.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        return fname, tensor

# ------------------------
# Schedule Helper
# ------------------------

def make_mixed_schedule(T=1000, N=DDIM_STEPS, p=POWER_P, fine_cutoff=FINE_CUTOFF):
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
    dataloader: DataLoader,
    save_dir: str,
    guidance_scale: float
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    # enable TF32 on Ampere+ for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    diffusion = Diffusion(device)
    betas = diffusion.beta.to(device).half()
    alpha_cumprod = diffusion.alpha_cumprod.to(device).half()
    T = diffusion.timesteps
    schedule = make_mixed_schedule(T=T)

    vae = vae.to(device).eval()
    unet = unet.to(device).eval()
    controlnet = controlnet.to(device).eval()
    dr_module = dr_module.to(device).eval()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        controlnet = nn.DataParallel(controlnet)
        dr_module = nn.DataParallel(dr_module)

    os.makedirs(save_dir, exist_ok=True)
    volume_start = time.time()

    for batch_idx, (names, imgs) in enumerate(dataloader, start=1):
        batch_start = time.time()
        imgs = imgs.to(device).half()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            control_inputs, _ = dr_module(imgs)
            mu, logvar = vae.encode(imgs)

            # Potentially do
            # std = (0.5 * logvar).exp()
            # z = mu + std * torch.randn_like(mu)
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

                # DDIM update
                z = ((z - sqrt_one_minus_a_t * eps) / sqrt_a_t) * a_prev.sqrt() \
                    + (1 - a_prev).sqrt() * eps

            gen = vae.decode(z)

        out_np = gen.cpu().float().numpy() * 1000.0
        for i, fname in enumerate(names):
            np.save(os.path.join(save_dir, fname), out_np[i].squeeze(0))
        print(f"Vol {os.path.basename(save_dir)} batch {batch_idx} in {time.time() - batch_start:.2f}s")

    print(f"Vol {os.path.basename(save_dir)} done in {time.time() - volume_start:.2f}s")

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
        ds = CBCTDatasetNPY(cbct_folder, transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        predict_volume(vae, unet, controlnet, dr_module, loader, save_folder, GUIDANCE_SCALE)
    print("All volumes processed.")
