#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    """Pad and resize a single 2D slice."""
    arr = np.array(img_np)
    padded = np.pad(
        arr,
        ((PAD_T, PAD_B), (PAD_L, PAD_R)),
        mode="constant",
        constant_values=-1000
    ).astype(np.int16)
    return np.array(
        Image.fromarray(padded)
             .resize((RES_W, RES_H), Image.BILINEAR)
    )

def crop_back(arr256):
    """Undo the padding/resizing to get back to original crop."""
    return arr256[
        TOP_CROP : RES_H - BOTTOM_CROP,
        LEFT_CROP : RES_W - RIGHT_CROP
    ]

def load_volume(dirpath, vidx, needs_transform=False):
    """
    Load a stack of .npy slices as a 3D volume.
    Optionally apply pad+resize transform to each slice.
    """
    pattern = os.path.join(dirpath, f"volume-{vidx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    sls = []
    for p in files:
        sl = np.load(p)
        if needs_transform:
            sl = apply_transform(sl)
        sl = crop_back(sl)
        sls.append(sl)
    vol = np.stack(sls, axis=0)
    print(f"Loaded {dirpath} → shape {vol.shape}")
    return vol

def extract_slice(vol, orientation, idx):
    """
    Extract a 2D slice from a 3D volume in the given orientation.
    orientation: 'axial', 'coronal', or 'sagittal'
    """
    if orientation == "axial":
        return np.fliplr(vol[idx, :, :])
    elif orientation == "coronal":
        return np.fliplr(np.flipud(vol[:, idx, :]))
    elif orientation == "sagittal":
        return np.flipud(vol[:, :, idx])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

def resize256(slice2d):
    """Resize a 2D NumPy array to 256×256 using bilinear interpolation."""
    return np.array(
        Image.fromarray(slice2d.astype(np.int16))
             .resize((256, 256), Image.BILINEAR)
    )

def plot_ct_cbct_axial(volume_idx, slice_idx):
    """
    Load CT and CBCT volumes for the given index,
    extract the central axial slice, and plot them side-by-side.
    """
    # ─── directories ─────────────────────────────────────────────────────────────
    ct_dir   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_dir = os.path.expanduser("~/thesis/training_data/CBCT/256/test")

    # ─── load volumes ────────────────────────────────────────────────────────────
    ct_vol   = load_volume(ct_dir,   volume_idx, needs_transform=True)
    cbct_vol = load_volume(cbct_dir, volume_idx, needs_transform=True)

    # ─── pick central axial index ───────────────────────────────────────────────
    idx = 50
    z_ct   = slice_idx
    z_cbct = slice_idx

    # ─── extract & resize axial slices ──────────────────────────────────────────
    ct_axial   = resize256(extract_slice(ct_vol,   "axial", z_ct))
    cbct_axial = resize256(extract_slice(cbct_vol, "axial", z_cbct))

    # ─── plot 1×2 ────────────────────────────────────────────────────────────────
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"]   = 14
    vmin, vmax = -400, 400
    vmin, vmax = -1000, 1000

    fig, (ax_ct, ax_cbct) = plt.subplots(
        1, 2,
        figsize=(8, 4),
        constrained_layout=True
    )

    # CT on the left
    ax_ct.imshow(ct_axial, cmap="gray", vmin=vmin, vmax=vmax)
    ax_ct.axis("off")
    ax_ct.set_title("CT", pad=6)

    # CBCT on the right
    ax_cbct.imshow(cbct_axial, cmap="gray", vmin=vmin, vmax=vmax)
    ax_cbct.axis("off")
    ax_cbct.set_title("CBCT", pad=6)

    # ─── save or show ───────────────────────────────────────────────────────────
    out = os.path.expanduser("~/thesis/figures/ct_cbct_axial.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved figure to {out}")
    plt.show()

if __name__ == "__main__":
    # Change the volume_idx as needed
    idx = 0
    while True:
        plot_ct_cbct_axial(volume_idx=33, slice_idx=idx)
        idx += 10

