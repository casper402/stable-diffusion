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
from quick_loop.unetConcatControlPACA import load_unet_concat_control_paca
from quick_loop.unetConditional import load_cond_unet, UNetCrossAttention
from torchvision.transforms import InterpolationMode


# ------------------------
# Configuration Variables
# ------------------------
CBCT_DIR = '../training_data/CBCT/490/test'
LIVER_DIR = '../training_data/liver/test'
TUMOR_DIR = '../training_data/liver/test'
VOLUME_INDICES = [3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129]
OUT_DIR = '../predictions/conditional_unet'

GUIDANCE_SCALE = 1.0
ALPHA_A = 0.2         # Mixing weight for CBCT signal at t0
BATCH_SIZE = 32       # tune as needed
# DDIM / schedule parameters: reduce steps for faster inference
DDIM_STEPS = 40       # total coarse sampling steps
POWER_P = 2.0         # power-law exponent for smoothing
FINE_CUTOFF = 9       # switch to single-step updates at t<=9 (last 10 steps)
STEP_SIZE = 20

SEGMENTATION_PATH = 'segmentation_controlnet_new_loss'
MODELS_PATH = 'unet_concat'
UNET_SAVE_PATH = os.path.join(MODELS_PATH, 'unet_v2.pth')
MODELS_PATH = '../best_model_v7'
VAE_SAVE_PATH = os.path.join(MODELS_PATH, 'vae_joint_vae.pth')
MODELS_PATH = 'unet_concat_control_paca'
PACA_LAYERS_SAVE_PATH = os.path.join(MODELS_PATH, 'paca_layers.pth')
CONTROLNET_SAVE_PATH = os.path.join(MODELS_PATH, 'controlnet_v2.pth')
DEGRADATION_REMOVAL_SAVE_PATH = os.path.join(MODELS_PATH, 'dr_module_v2.pth')
CONTROLNET_SEG_SAVE_PATH = os.path.join(SEGMENTATION_PATH, 'controlnet_seg.pth')
DEGRADATION_REMOVAL_SEG_SAVE_PATH = os.path.join(SEGMENTATION_PATH, 'dr_module_seg.pth')

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
# Dataset for CBCT + Seg slices
# ------------------------
class CBCTSegmentationDatasetNPY(Dataset):
    def __init__(self, cbct_dir: str, tumor_dir: str, liver_dir: str, cbct_transform=None, seg_transform=None):
        self.cbct_files = sorted([f for f in os.listdir(cbct_dir) if f.endswith('.npy')])
        self.cbct_dir = cbct_dir
        self.tumor_dir = tumor_dir
        self.liver_dir = liver_dir
        self.cbct_transform = cbct_transform
        self.seg_transofrm = seg_transform

    def __len__(self):
        return len(self.cbct_files)

    def __getitem__(self, idx):
        fname = self.cbct_files[idx]
        cbct = np.load(os.path.join(self.cbct_dir, fname)).astype(np.float32) / 1000.0
        
        liver_path = os.path.join(self.liver_dir, fname)
        tumor_path = os.path.join(self.tumor_dir, fname)
        if os.path.exists(liver_path):
            liver = np.load(liver_path).astype(np.float32)
        else:
            print(f"no liver path: {liver_path}")
            liver = np.zeros(cbct.shape, dtype=np.float32)
        if os.path.exists(tumor_path):
            tumor = np.load(tumor_path).astype(np.float32)
        else:
            print(f"no tumor path: {tumor_path}")
            tumor = np.zeros(cbct.shape, dtype=np.float32)

        segmentation_map = liver - tumor
        segmentation_map_tensor = torch.from_numpy(segmentation_map).unsqueeze(0)
        cbct_tensor = torch.from_numpy(cbct).unsqueeze(0)

        if self.cbct_transform:
            cbct_tensor = self.cbct_transform(cbct_tensor)
        if self.seg_transofrm:
            segmentation_map_tensor = self.seg_transofrm(segmentation_map_tensor)
        return fname, cbct_tensor, segmentation_map_tensor

# ------------------------
# Schedule Helpers
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

def make_linear_schedule(T: int, step_size: int = 10) -> np.ndarray:
    """
    Create a schedule of timesteps from T down to 0,
    stepping by `step_size` each time.
    """
    ts = np.arange(T, -1, -step_size, dtype=int)
    if ts[-1] != 0:
        ts = np.concatenate([ts, [0]])
    return ts

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

    diffusion = Diffusion(device)
    betas = diffusion.beta.to(device)
    alpha_cumprod = diffusion.alpha_cumprod.to(device)
    T = diffusion.timesteps

    # Choose sampling schedule
    schedule = make_linear_schedule(T=T-1, step_size=STEP_SIZE)
    t0 = int(schedule[0])

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
        imgs = imgs.to(device)
        with torch.inference_mode():
            control_inputs, _ = dr_module(imgs)
            mu, logvar = vae.encode(imgs)

            # -----------------------------------------------------------------
            # New init: mix CBCT signal (mu) with noise at t0 using ALPHA_A
            # -----------------------------------------------------------------
            alpha_bar_t0 = alpha_cumprod[t0]
            alpha_eff    = ALPHA_A * alpha_bar_t0
            s_alpha      = torch.sqrt(alpha_eff)
            s_noise      = torch.sqrt(1.0 - alpha_eff)
            z        = torch.randn_like(mu)
            # z            = s_alpha * mu + s_noise * noise

            # PACA control diffusion loop
            for i in range(len(schedule) - 1):
                t, t_prev = int(schedule[i]), int(schedule[i + 1])
                t_tensor   = torch.full((z.size(0),), t, device=device, dtype=torch.long)
                down_res, mid_res = controlnet(z, control_inputs, t_tensor)
                eps = unet(z, t_tensor, down_res, mid_res)

                a_t      = alpha_cumprod[t]
                a_prev   = alpha_cumprod[t_prev]
                sqrt_at  = a_t.sqrt()
                sqrt_omt = (1 - a_t).sqrt()

                # DDIM update rule
                z = ((z - sqrt_omt * eps) / sqrt_at) * a_prev.sqrt() + (1 - a_prev).sqrt() * eps

            gen = vae.decode(z)

        out_np = gen.cpu().float().numpy() * 1000.0
        for i, fname in enumerate(names):
            np.save(os.path.join(save_dir, fname), out_np[i].squeeze(0))
        print(f"Vol {os.path.basename(save_dir)} batch {batch_idx} in {time.time() - batch_start:.2f}s")

    print(f"Vol {os.path.basename(save_dir)} done in {time.time() - volume_start:.2f}s")

def conditional_unet_predict_volume(
    vae, unet,
    dataloader: DataLoader,
    save_dir: str,
    guidance_scale: float
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diffusion = Diffusion(device)
    betas = diffusion.beta.to(device)
    alpha_cumprod = diffusion.alpha_cumprod.to(device)
    T = diffusion.timesteps

    # Choose sampling schedule
    schedule = make_linear_schedule(T=T-1, step_size=STEP_SIZE)
    t0 = int(schedule[0])

    vae = vae.to(device).eval()
    unet = unet.to(device).eval()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)

    os.makedirs(save_dir, exist_ok=True)
    volume_start = time.time()

    for batch_idx, (names, imgs) in enumerate(dataloader, start=1):
        batch_start = time.time()
        imgs = imgs.to(device)
        with torch.inference_mode():
            mu, logvar = vae.encode(imgs)
            # -----------------------------------------------------------------
            # New init: mix CBCT signal (mu) with noise at t0 using ALPHA_A
            # -----------------------------------------------------------------
            alpha_bar_t0 = alpha_cumprod[t0]
            alpha_eff    = ALPHA_A * alpha_bar_t0
            s_alpha      = torch.sqrt(alpha_eff)
            s_noise      = torch.sqrt(1.0 - alpha_eff)
            z        = torch.randn_like(mu)
            # z            = s_alpha * mu + s_noise * noise

            # PACA control diffusion loop
            for i in range(len(schedule) - 1):
                t, t_prev = int(schedule[i]), int(schedule[i + 1])
                t_tensor   = torch.full((z.size(0),), t, device=device, dtype=torch.long)
                eps = unet(z, mu, t_tensor)

                a_t      = alpha_cumprod[t]
                a_prev   = alpha_cumprod[t_prev]
                sqrt_at  = a_t.sqrt()
                sqrt_omt = (1 - a_t).sqrt()

                # DDIM update rule
                z = ((z - sqrt_omt * eps) / sqrt_at) * a_prev.sqrt() + (1 - a_prev).sqrt() * eps

            gen = vae.decode(z)

        out_np = gen.cpu().float().numpy() * 1000.0
        for i, fname in enumerate(names):
            np.save(os.path.join(save_dir, fname), out_np[i].squeeze(0))
        print(f"Vol {os.path.basename(save_dir)} batch {batch_idx} in {time.time() - batch_start:.2f}s")

    print(f"Vol {os.path.basename(save_dir)} done in {time.time() - volume_start:.2f}s")

def predict_concat_control_paca(
    vae, unet, controlnet, dr_module,
    dataloader: DataLoader,
    save_dir: str,
    guidance_scale: float
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diffusion = Diffusion(device)
    betas = diffusion.beta.to(device)
    alpha_cumprod = diffusion.alpha_cumprod.to(device)
    T = diffusion.timesteps

    # Choose sampling schedule
    schedule = make_linear_schedule(T=T-1, step_size=STEP_SIZE)
    t0 = int(schedule[0])

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
        imgs = imgs.to(device)
        with torch.inference_mode():
            control_inputs, _ = dr_module(imgs)
            mu, logvar = vae.encode(imgs)

            # -----------------------------------------------------------------
            # New init: mix CBCT signal (mu) with noise at t0 using ALPHA_A
            # -----------------------------------------------------------------
            alpha_bar_t0 = alpha_cumprod[t0]
            alpha_eff    = ALPHA_A * alpha_bar_t0
            s_alpha      = torch.sqrt(alpha_eff)
            s_noise      = torch.sqrt(1.0 - alpha_eff)
            z        = torch.randn_like(mu)
            # z            = s_alpha * mu + s_noise * noise

            # PACA control diffusion loop
            for i in range(len(schedule) - 1):
                t, t_prev = int(schedule[i]), int(schedule[i + 1])
                t_tensor   = torch.full((z.size(0),), t, device=device, dtype=torch.long)
                down_res, mid_res = controlnet(z, control_inputs, t_tensor)
                eps = unet(z, mu, t_tensor, down_res, mid_res)

                a_t      = alpha_cumprod[t]
                a_prev   = alpha_cumprod[t_prev]
                sqrt_at  = a_t.sqrt()
                sqrt_omt = (1 - a_t).sqrt()

                # DDIM update rule
                z = ((z - sqrt_omt * eps) / sqrt_at) * a_prev.sqrt() + (1 - a_prev).sqrt() * eps

            gen = vae.decode(z)

        out_np = gen.cpu().float().numpy() * 1000.0
        for i, fname in enumerate(names):
            np.save(os.path.join(save_dir, fname), out_np[i].squeeze(0))
        print(f"Vol {os.path.basename(save_dir)} batch {batch_idx} in {time.time() - batch_start:.2f}s")

    print(f"Vol {os.path.basename(save_dir)} done in {time.time() - volume_start:.2f}s")

def segmentation_predict_volume(
    vae, unet, controlnet, dr_module, controlnet_seg, dr_module_seg,
    dataloader: DataLoader,
    save_dir: str,
    guidance_scale: float,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diffusion = Diffusion(device)
    betas = diffusion.beta.to(device)
    alpha_cumprod = diffusion.alpha_cumprod.to(device)
    T = diffusion.timesteps

    # Choose sampling schedule
    schedule = make_linear_schedule(T=T-1, step_size=STEP_SIZE)
    t0 = int(schedule[0])

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

    for batch_idx, (names, imgs, segmentation) in enumerate(dataloader, start=1):
        batch_start = time.time()
        imgs = imgs.to(device)
        segmentation = segmentation.to(device)
        with torch.inference_mode():
            control_inputs, _ = dr_module(imgs)
            controlnet_input_seg, _ = dr_module_seg(segmentation)

            mu, logvar = vae.encode(imgs)

            # -----------------------------------------------------------------
            # New init: mix CBCT signal (mu) with noise at t0 using ALPHA_A
            # -----------------------------------------------------------------
            alpha_bar_t0 = alpha_cumprod[t0]
            alpha_eff    = ALPHA_A * alpha_bar_t0
            s_alpha      = torch.sqrt(alpha_eff)
            s_noise      = torch.sqrt(1.0 - alpha_eff)
            z        = torch.randn_like(mu)
            # z            = s_alpha * mu + s_noise * noise

            # PACA control diffusion loop
            for i in range(len(schedule) - 1):
                t, t_prev = int(schedule[i]), int(schedule[i + 1])
                t_tensor   = torch.full((z.size(0),), t, device=device, dtype=torch.long)
                down_res, mid_res = controlnet(z, control_inputs, t_tensor)
                seg_down_res, seg_mid_res = controlnet_seg(z, controlnet_input_seg, t_tensor)
                eps = unet(z, t_tensor, down_res, mid_res, seg_down_res, seg_mid_res)

                a_t      = alpha_cumprod[t]
                a_prev   = alpha_cumprod[t_prev]
                sqrt_at  = a_t.sqrt()
                sqrt_omt = (1 - a_t).sqrt()

                # DDIM update rule
                z = ((z - sqrt_omt * eps) / sqrt_at) * a_prev.sqrt() + (1 - a_prev).sqrt() * eps

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
    seg_transform = transforms.Compose([
        transforms.Pad((0, 64, 0, 64), fill=0),
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST_EXACT),
    ])
    vae = load_vae(VAE_SAVE_PATH)
    #unet = load_unet_control_paca(UNET_SAVE_PATH, PACA_LAYERS_SAVE_PATH)
    controlnet = load_controlnet(CONTROLNET_SAVE_PATH)
    dr_module = load_degradation_removal(DEGRADATION_REMOVAL_SAVE_PATH)
    #controlnet_seg = load_controlnet(CONTROLNET_SEG_SAVE_PATH)
    #dr_module_seg = load_degradation_removal(DEGRADATION_REMOVAL_SEG_SAVE_PATH)
    unet = load_unet_concat_control_paca(UNET_SAVE_PATH, PACA_LAYERS_SAVE_PATH)

    for vol in VOLUME_INDICES:
        cbct_folder = os.path.join(CBCT_DIR, f"volume-{vol}")
        save_folder = os.path.join(OUT_DIR, f"volume-{vol}")
        ds = CBCTDatasetNPY(cbct_folder, transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        #segmentation_predict_volume(vae, unet, controlnet, dr_module, controlnet_seg, dr_module_seg, loader, save_folder, GUIDANCE_SCALE)
        predict_concat_control_paca(vae, unet, controlnet, dr_module, loader, save_folder, GUIDANCE_SCALE)
    print("All volumes processed.")
