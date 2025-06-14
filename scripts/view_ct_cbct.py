import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# ───── constants ──────────────────────────────────────────────────────────────
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))

def apply_transform(img_np):
    arr = np.array(img_np)
    padded = np.pad(arr,
                    ((PAD_T,PAD_B),(PAD_L,PAD_R)),
                    mode="constant", constant_values=-1000).astype(np.int16)
    return np.array(Image.fromarray(padded)
                    .resize((RES_W,RES_H), Image.BILINEAR))

def crop_back(arr256):
    return arr256[TOP_CROP:RES_H-BOTTOM_CROP,
                  LEFT_CROP:RES_W-RIGHT_CROP]

def load_volume(dirpath, vidx, needs_transform=False):
    pattern = os.path.join(dirpath, f"volume-{vidx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files in {pattern}")
    sls = []
    for p in files:
        sl = np.load(p)
        if needs_transform:
            sl = apply_transform(sl)
        sl = crop_back(sl)
        sls.append(sl)
    vol = np.stack(sls, 0)
    print(f"Loaded {dirpath} → {vol.shape}")
    return vol

def extract_slice(vol, orientation, idx):
    if orientation == "axial":
        return np.fliplr(vol[idx])
    elif orientation == "coronal":
        return np.fliplr(np.flipud(vol[:, idx, :]))
    elif orientation == "sagittal":
        return np.flipud(vol[:, :, idx])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

def resize256(slice2d):
    return np.array(Image.fromarray(slice2d.astype(np.int16))
                    .resize((256,256), Image.BILINEAR))

def plot_ct_cbct_3x2(volume_idx):
    # Directories
    ct_dir   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_dir = os.path.expanduser("~/thesis/training_data/CBCT/256/test")

    # 1) Load volumes
    ct_vol   = load_volume(ct_dir,   volume_idx, needs_transform=True)
    cbct_vol = load_volume(cbct_dir, volume_idx, needs_transform=True)

    # 2) Choose central slices for each
    def central_indices(vol):
        Z,H,W = vol.shape
        return Z//2, H//2, W//2
    
    indices = 150, 130, 70

    ct_idxs    = indices
    cbct_idxs  = indices

    # 3) Extract & resize for each orientation
    orients = ['axial','coronal','sagittal']
    ct_slices   = [resize256(extract_slice(ct_vol,   o, idx))
                   for o,idx in zip(orients, ct_idxs)]
    cbct_slices = [resize256(extract_slice(cbct_vol, o, idx))
                   for o,idx in zip(orients, cbct_idxs)]

    # 4) Plot grid 2×3
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"]   = 14
    vmin,vmax = -400,400

    fig, axes = plt.subplots(2, 3,
                             figsize=(3*3, 2*3),
                             constrained_layout=True)

    # First row: CT
    for j, sl in enumerate(ct_slices):
        ax = axes[0,j]
        ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if j == 0:
            ax.text(
                -0.06, 0.5, f"CT",
                va="center", ha="center",
                rotation="vertical",
                transform=ax.transAxes
            )
        ax.set_title(orients[j].capitalize(), pad=6)

    # Second row: CBCT Q=256
    for j, sl in enumerate(cbct_slices):
        ax = axes[1,j]
        ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if j == 0:
            ax.text(
                -0.06, 0.5, f"CBCT",
                va="center", ha="center",
                rotation="vertical",
                transform=ax.transAxes
            )

    # 5) Save or show
    out = os.path.expanduser("~/thesis/figures/ct_cbct_3x2.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved to {out}")
    plt.show()

if __name__ == "__main__":
    plot_ct_cbct_3x2(volume_idx=8)
