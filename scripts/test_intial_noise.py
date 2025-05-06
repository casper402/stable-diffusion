#!/usr/bin/env python3
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from models.diffusion import Diffusion
from quick_loop.vae import load_vae

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CT_DIR       = "../../training_data/CT/train"
CBCT_DIR     = "../../training_data/CBCT/train"
VOLUME       = 1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA_A_LIST = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_IMAGES   = 10  # only preprocess first 5 slices
VAE_PATH     = "../vae_joint_v2.pth"
VAE_PATH     = "../vaeV6.pth"
# ─────────────────────────────────────────────────────────────────────────────

# ─── pad→resize→normalize (no cropping) ───────────────────────────────────────
pad_resize = transforms.Compose([
    transforms.Pad((0, 64, 0, 64), fill=-1000),
    transforms.Resize((256, 256)),
])

def preprocess(np_img):
    pil = Image.fromarray(np_img)
    out = pad_resize(pil)
    arr = np.array(out, dtype=np.float32) / 1000.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W) in [-1,1]


def load_and_prepare(ct_folder, cbct_folder, volume_idx, vae, diffusion, max_slices):
    """
    Load up to max_slices CT .npy slices, preprocess, encode once,
    generate a fixed noise map per slice and compute baseline z_ct,
    encode corresponding CBCT slice, return list of
    (fname, mu_cb, noise_ct, z_ct_base) on DEVICE.
    """
    ct_path   = os.path.join(ct_folder, f"volume-{volume_idx}")
    cbct_path = os.path.join(cbct_folder, f"volume-{volume_idx}")
    files = sorted(f for f in os.listdir(ct_path) if f.endswith(".npy"))[:max_slices]
    results = []
    t_idx     = diffusion.timesteps - 1

    for f in files:
        # CT encoding and baseline noise
        arr_ct   = np.load(os.path.join(ct_path, f)).astype(np.float32)
        x_ct     = preprocess(arr_ct).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu_ct, _ = vae.encode(x_ct)
        noise_ct  = torch.randn_like(mu_ct, device=DEVICE)
        z_ct_base = diffusion.add_noise(mu_ct, t_idx, noise_ct)

        # CBCT encoding
        arr_cb    = np.load(os.path.join(cbct_path, f)).astype(np.float32)
        x_cb      = preprocess(arr_cb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu_cb, _ = vae.encode(x_cb)

        results.append((f, mu_cb, noise_ct, z_ct_base))
    return results


def main():
    # 1) Load VAE & diffusion scheduler
    vae       = load_vae(VAE_PATH).to(DEVICE).eval()
    diffusion = Diffusion(DEVICE)

    # 2) Prepare data
    data = load_and_prepare(CT_DIR, CBCT_DIR, VOLUME, vae, diffusion, NUM_IMAGES)

    # 3) Pure-noise baseline distances to CT noisy latent
    noise_dists = [(noise - z_ct).norm().item() for _, _, noise, z_ct in data]
    mean_noise  = float(np.mean(noise_dists))
    print(f"Pure noise baseline (mean L2(noise, z_ct_base)): {mean_noise:.2f}")

    # 4) Precompute bar-alpha at t=999
    t_idx     = diffusion.timesteps - 1
    alpha_bar = diffusion.alpha_cumprod[t_idx].item()

    # 5) Sweep alpha_a, compute CBCT-CT distances
    results = []
    for alpha_a in ALPHA_A_LIST:
        diffs = []
        alpha_eff = alpha_a * alpha_bar
        s_alpha   = torch.sqrt(torch.tensor(alpha_eff, device=DEVICE))
        s_one     = torch.sqrt(torch.tensor(1 - alpha_eff, device=DEVICE))

        for fname, mu_cb, noise_ct, z_ct_base in data:
            # noisy CBCT latent using the SAME noise_ct
            z_cb = s_alpha * mu_cb + s_one * noise_ct
            # distance to CT baseline latent
            diffs.append((z_cb - z_ct_base).norm().item())

        results.append((alpha_a, float(np.mean(diffs))))

    # 6) Print summary
    print("\n alpha_a | mean L2(z_cb, z_ct_base) ")
    print("------------------------------------")
    for alpha_a, m_diff in results:
        print(f" {alpha_a:7.2f} | {m_diff:23.2f}")

if __name__ == "__main__":
    main()
