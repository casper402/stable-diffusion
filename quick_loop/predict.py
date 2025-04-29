import os
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from models.diffusion import Diffusion
from quick_loop.vae import load_vae, train_vae
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from quick_loop.unetControlPACA import (
    load_unet_control_paca,
    train_dr_control_paca,
    test_dr_control_paca
)

# ------------------------
# Configuration Variables
# ------------------------
CBCT_DIR = '../training_data/CBCT/test/'
# Possible volumes: 3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129
# Volumes done: 3
VOLUME_IDX = 8
OUT_DIR = '../predictions/'
GUIDANCE_SCALES = [1.0]
MODELS_PATH = 'controlnet_training/v2/'
VAE_SAVE_PATH = MODELS_PATH + 'vae.pth'
UNET_SAVE_PATH = MODELS_PATH + 'unet.pth'
PACA_LAYERS_SAVE_PATH = MODELS_PATH + 'paca_layers.pth'
CONTROLNET_SAVE_PATH = MODELS_PATH + 'controlnet.pth'
DEGRADATION_REMOVAL_SAVE_PATH = MODELS_PATH + 'dr_module.pth'

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


def predict_volume(
    vae, unet, controlnet, dr_module,
    cbct_slices,
    save_dir: str,
    guidance_scales
):
    """
    Run inference on each CBCT slice and save predicted slices as .npy under save_dir.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(device)

    vae.to(device).eval()
    unet.to(device).eval()
    controlnet.to(device).eval()
    dr_module.to(device).eval()

    os.makedirs(save_dir, exist_ok=True)

    for cbct_name, cbct_tensor in cbct_slices:
        cbct = cbct_tensor.unsqueeze(0).to(device)
        print(f"Starting to predict for {cbct_name}")

        with torch.no_grad():
            control_input, _ = dr_module(cbct)
            z_t = torch.randn_like(vae.encode(cbct)[0])
            T = diffusion.timesteps

            for guidance_scale in guidance_scales:
                for t_int in range(T - 1, -1, -1):
                    t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

                    down_res, mid_res = controlnet(z_t, control_input, t)
                    pred_cond = unet(z_t, t, down_res, mid_res)
                    pred_uncond = unet(z_t, t, None, None)
                    pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                    beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
                    alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
                    alpha_cum = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
                    coef = beta_t / torch.sqrt(1.0 - alpha_cum)
                    mean = torch.sqrt(1.0 / alpha_t) * (z_t - coef * pred_noise)

                    if t_int > 0:
                        noise = torch.randn_like(z_t)
                        z_t = mean + torch.sqrt(beta_t) * noise
                    else:
                        z_t = mean

                z_0 = z_t
                gen = vae.decode(z_0)[0]
                output = (gen / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze(0)

                out_name = cbct_name
                save_path = os.path.join(save_dir, out_name)
                np.save(save_path, output)

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
    unet = load_unet_control_paca(
        unet_save_path=UNET_SAVE_PATH,
        paca_save_path=PACA_LAYERS_SAVE_PATH
    )
    controlnet = load_controlnet(save_path=CONTROLNET_SAVE_PATH)
    dr_module = load_degradation_removal(save_path=DEGRADATION_REMOVAL_SAVE_PATH)

    predict_volume(
        vae, unet, controlnet, dr_module,
        cbct_slices,
        save_folder,
        GUIDANCE_SCALES
    )
