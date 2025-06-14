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

SLICE_RANGES = {
    3: None,    8: (0,354),  12: (0,320), 26: None,
    32: (69,269), 33: (59,249), 35: (91,268),
    54: (0,330), 59:(0,311), 61:(0,315),
    106: None, 116: None, 129: (5,346),
}

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
    vol = np.stack(sls,0)
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

def plot_cbct_views(volume_idx, slice_idx = None):
    quals = [490, 256, 128, 64, 32]

    # 1) Load one sample to get Z,H,W and slice‐range
    sample = load_volume(
        os.path.expanduser("~/thesis/training_data/CBCT/256/test"),
        volume_idx, True
    )
    Z, H, W = sample.shape

    # 2) Pick slice indices
    if volume_idx in SLICE_RANGES and SLICE_RANGES[volume_idx]:
        lb,ub = SLICE_RANGES[volume_idx]; ub = min(ub, Z-1)
    else:
        lb,ub = 0, Z-1
    axial_idx    = 150 if lb<=150<=ub else random.randint(lb,ub)
    coronal_idx  = 130
    sagittal_idx = 70
    if slice_idx is not None:
        coronal_idx  = slice_idx
        sagittal_idx = slice_idx

    print(f"Using slices → axial={axial_idx}, coronal={coronal_idx}, sagittal={sagittal_idx}")

    # 3) For each quality, load and extract all three orientations
    cbct_slices = {}
    for q in quals:
        cb_dir = os.path.expanduser(f"~/thesis/training_data/CBCT/{q}/test")
        vol = load_volume(cb_dir, volume_idx, True)
        # extract and resize
        axial = resize256(extract_slice(vol, 'axial', axial_idx))
        coronal = resize256(extract_slice(vol, 'coronal', coronal_idx))
        sagittal = resize256(extract_slice(vol, 'sagittal', sagittal_idx))
        # store both full-range and windowed views
        cbct_slices[q] = {
            'axial_full': axial,
            'axial': axial,
            'coronal': coronal,
            'sagittal': sagittal,
        }

    # 4) Plot a grid: 5 rows (Q), 4 cols (axial_full, axial, coronal, sagittal)
    plt.rcParams["font.family"]    = "serif"
    plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
    plt.rcParams["font.size"]      = 18
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 18
    cols = ['Axial [-1000,1000]', 'Axial [-400, 400]', 'Coronal [-400, 400]', 'Sagittal [-400, 400]']
    vlims = {
        'axial_full': (-1000, 1000),
        'axial': (-400, 400),
        'coronal': (-400, 400),
        'sagittal': (-400, 400),
    }

    fig, axes = plt.subplots(
        len(quals), 4,
        figsize=(4*3, len(quals)*3),
        constrained_layout=True
    )
    for i, q in enumerate(quals):
        for j, orient in enumerate(['axial_full', 'axial', 'coronal', 'sagittal']):
            ax = axes[i, j]
            vmin, vmax = vlims[orient]
            ax.imshow(cbct_slices[q][orient], cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if i == 0:
                ax.set_title(cols[j], pad=6)
            # if j == 0:
            #     ax.text(
            #         -0.06, 0.5, f"Q={q}",
            #         va="center", ha="center",
            #         rotation="vertical",
            #         transform=ax.transAxes
            #     )

    # 5) Save or show
    out = os.path.expanduser("~/thesis/figures/cbct_4views_per_quality.pdf")
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved to {out}")
    plt.show()

if __name__ == "__main__":
    plot_cbct_views(volume_idx=8)
