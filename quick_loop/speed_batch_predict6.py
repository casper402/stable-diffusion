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
PREDICT_CLINIC = False

CBCT_DIR = '../training_data/scaled-490/'
CBCT_CLINIC_DIR = '../training_data/clinic/'
# VOLUME_INDICES = [3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129]
VOLUME_INDICES = [8]

GUIDANCE_SCALE = 1.0
ALPHA_A = 0.2         # Mixing weight for CBCT signal at t0
BATCH_SIZE = 16 # Can probably be 32
POWER_P = 2.0

PREPROCESS = "linear" # linear or tanh

# MODELS_PATH = 'controlnet_v2_inference_v2/'
# MODELS_PATH = 'controlnet_v3'
# MODELS_PATH = 'controlnet_from_unet_trained_after_joint'
# MODELS_PATH = 'controlnet_v4'
# MODELS_PATH = 'non-linear-vae-controlnet'
# MODELS_PATH = 'non-linear-vae-controlnet-5'
MODELS_PATH = 'controlnet_v7-data-augmentation' # BEST MODEL

# VAE_SAVE_PATH = os.path.join(MODELS_PATH, 'vae_joint_vae_nonlinear.pth')
# UNET_SAVE_PATH = os.path.join(MODELS_PATH, 'unet_joint_unet_nonlinear.pth')
VAE_SAVE_PATH = os.path.join(MODELS_PATH, 'vae_joint_vae.pth')
UNET_SAVE_PATH = os.path.join(MODELS_PATH, 'unet_joint_unet.pth')
PACA_LAYERS_SAVE_PATH = os.path.join(MODELS_PATH, 'paca_layers.pth')
CONTROLNET_SAVE_PATH = os.path.join(MODELS_PATH, 'controlnet.pth')
DEGRADATION_REMOVAL_SAVE_PATH = os.path.join(MODELS_PATH, 'dr_module.pth')

# ------------------------
# NaN Assertion Helper
# ------------------------
def assert_no_nan(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN encountered in {name}")

# ------------------------
# Dataset for CBCT slices
# ------------------------
class CBCTDatasetNPY(Dataset):
    def __init__(self, volume_dir: str, transform=None, preprocess="linear"):
        self.files = sorted([f for f in os.listdir(volume_dir) if f.endswith('.npy')])
        self.volume_dir = volume_dir
        self.transform = transform
        assert preprocess in ["linear", "tanh"]
        self.preprocess = preprocess
        print("using preprocess:", preprocess)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        arr = np.load(os.path.join(self.volume_dir, fname)).astype(np.float32)

        if self.preprocess == "linear":
            arr /= 1000.0
        elif self.preprocess == "tanh":
            # apply a "soft window" around ±150 HU:
            #  - inside ±150 HU it's almost linear (tanh(x/150) ≈ x/150 for |x|≲100),
            #  - beyond ±150 HU things smoothly compress toward ±1.
            arr = np.tanh(arr / 350.0)

        tensor = torch.from_numpy(arr).unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        return fname, tensor

def postprocess_linear(CT):
    """
    Scale and clip CT Hounsfield units for visualization.
    """
    scaled = torch.round(CT * 1000)
    clipped = torch.clamp(scaled, -1000, 1000)
    return clipped.cpu().numpy()

def postprocess_tanh(CT, eps: float = 1e-4):
    """
    Invert tanh-based preprocessing (tanh(HU/150)) and
    prepare for display (round + clip to [-1000, 1000]).
    """
    # 1. Clamp into (–1,1) so atanh stays finite
    # x = torch.clamp(CT, -1 + eps, 1 - eps)
    x = torch.clamp(CT, -1, 1)

    # 2. Inverse tanh via torch.atanh, then rescale
    hu = torch.atanh(x) * 150.0

    # 3. Round to integer HUs and clamp to [-1000,1000]
    hu = torch.round(hu)
    hu = torch.clamp(hu, -1000.0, 1000.0)

    return hu.cpu().numpy().astype(np.float32)

# ------------------------
# Schedule Helpers
# ------------------------
def make_power_schedule(T=1000, N=50, p=POWER_P):
    print("making power schedule with:", N, "steps")
    # 1) Generat e N+1 equally spaced indices from 0 to N
    idx = np.arange(N + 1)
    # 2) Apply the power-law formula (1 - (i/N)^p) * T
    raw = (1 - (idx / N) ** p) * T
    # 3) Cast to int and reverse so that it goes from T down to 0
    ts = raw.astype(int)[::-1]
    return ts

def make_linear_schedule(T: int, step_size: int = 10) -> np.ndarray:
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
    guidance_scale: float,
    STEPS
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    diffusion = Diffusion(device)
    betas = diffusion.beta.to(device).float()
    alpha_cumprod = diffusion.alpha_cumprod.to(device).float()
    T = diffusion.timesteps

    schedule = make_power_schedule(T=T-1, N=STEPS)

    # Move models to float32
    vae = vae.to(device).float().eval()
    unet = unet.to(device).float().eval()
    controlnet = controlnet.to(device).float().eval()
    dr_module = dr_module.to(device).float().eval()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet)
        controlnet = nn.DataParallel(controlnet)
        dr_module = nn.DataParallel(dr_module)

    os.makedirs(save_dir, exist_ok=True)
    volume_start = time.time()

    for batch_idx, (names, imgs) in enumerate(dataloader, start=1):
        batch_start = time.time()
        imgs = imgs.to(device).float()      # ← float32 input

        with torch.inference_mode():
            control_inputs, _ = dr_module(imgs)
            mu, logvar = vae.encode(imgs)
            assert_no_nan(mu,     "VAE mu")
            assert_no_nan(logvar, "VAE logvar")

            z = torch.randn_like(mu, device=device, dtype=torch.float32)
            assert_no_nan(z, "initial z")

            for i in range(len(schedule) - 1):
                t, t_prev = int(schedule[i]), int(schedule[i+1])
                t_tensor = torch.full((z.size(0),), t, device=device, dtype=torch.long)

                down_res, mid_res = controlnet(z, control_inputs, t_tensor)
                eps = unet(z, t_tensor, down_res, mid_res)
                assert_no_nan(eps, f"eps at step {i} (t={t})")

                a_t    = alpha_cumprod[t]
                a_prev = alpha_cumprod[t_prev]
                sqrt_at  = a_t.sqrt()
                sqrt_omt = (1 - a_t).sqrt()

                z = ((z - sqrt_omt * eps) / sqrt_at) * a_prev.sqrt() + (1 - a_prev).sqrt() * eps
                assert_no_nan(z, f"z after step {i} (t={t})")

            gen = vae.decode(z)
            assert_no_nan(gen, "decoder output")

        if PREPROCESS == "linear":
            out_np = postprocess_linear(gen)
        elif PREPROCESS == "tanh":
            out_np = postprocess_tanh(gen)
        else:
            raise Exception("incorrect preprocess:", PREPROCESS)
        for i, fname in enumerate(names):
            np.save(os.path.join(save_dir, fname), out_np[i].squeeze(0))

        print(f"Vol {os.path.basename(save_dir)} batch {batch_idx} in {time.time() - batch_start:.2f}s")

    print(f"Vol {os.path.basename(save_dir)} done in {time.time() - volume_start:.2f}s")

# ------------------------
# Entry Points
# ------------------------
def predict_test_data(steps, i):
    transform = transforms.Compose([
        transforms.Pad((0, 64, 0, 64), fill=-1),
        transforms.Resize((256, 256)),
    ])
    vae        = load_vae(VAE_SAVE_PATH)
    unet       = load_unet_control_paca(UNET_SAVE_PATH, PACA_LAYERS_SAVE_PATH)
    controlnet = load_controlnet(CONTROLNET_SAVE_PATH)
    dr_module  = load_degradation_removal(DEGRADATION_REMOVAL_SAVE_PATH)

    OUT_DIR = f"../thesis-ready/490/best-model/ddim/power/{steps}-steps/{i}"

    print("using model from:", MODELS_PATH)
    print("saving in:", OUT_DIR) 

    for vol in VOLUME_INDICES:
        cbct_folder = os.path.join(CBCT_DIR, f"volume-{vol}")
        save_folder = os.path.join(OUT_DIR, f"volume-{vol}")
        ds     = CBCTDatasetNPY(cbct_folder, transform, preprocess=PREPROCESS)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        predict_volume(vae, unet, controlnet, dr_module, loader, save_folder, GUIDANCE_SCALE, steps)
    print("All volumes processed.")

def predict_clinic():
    clinic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    vae        = load_vae(VAE_SAVE_PATH)
    unet       = load_unet_control_paca(UNET_SAVE_PATH, PACA_LAYERS_SAVE_PATH)
    controlnet = load_controlnet(CONTROLNET_SAVE_PATH)
    dr_module  = load_degradation_removal(DEGRADATION_REMOVAL_SAVE_PATH)

    ds     = CBCTDatasetNPY(CBCT_CLINIC_DIR, clinic_transform, preprocess=PREPROCESS)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    print("ready to predict for:", CBCT_CLINIC_DIR)
    predict_volume(vae, unet, controlnet, dr_module, loader, OUT_DIR, GUIDANCE_SCALE)

if __name__ == '__main__':
    if PREDICT_CLINIC:
        predict_clinic()
    else:
        for steps in [1, 2, 5, 10, 25, 50]:
            for i in range(10):
                predict_test_data(steps, i)
