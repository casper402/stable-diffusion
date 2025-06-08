import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from evaluation import compute_mae, compute_rmse, compute_psnr, DATA_RANGE, ssim

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
    3: None, 8: (0,354), 12: (0,320), 26: None,
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

def extract_axial(vol, idx):
    return np.fliplr(vol[idx])

def resize256(slice2d):
    return np.array(Image.fromarray(slice2d.astype(np.int16))
                    .resize((256,256), Image.BILINEAR))

def plot_custom(volume_idx):
    quals = [490,256,128,64,32]

    plt.rcParams["font.family"]    = "serif"
    plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
    plt.rcParams["font.size"]      = 16
    plt.rcParams["axes.titlesize"] = 16

    # 1) pick axial idx from CBCT@256
    sample = load_volume(
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/256/test"),
        volume_idx, True
    )
    Z = sample.shape[0]
    if volume_idx in SLICE_RANGES and SLICE_RANGES[volume_idx]:
        lb,ub = SLICE_RANGES[volume_idx]; ub = min(ub, Z-1)
    else:
        lb,ub = 0, Z-1
    axial_idx = 150 if lb<=150<=ub else random.randint(lb,ub)
    print(f"Axial idx = {axial_idx} ({lb}–{ub})")

    # 2) load CT once
    ct_vol = load_volume(
        os.path.expanduser("/Users/Niklas/thesis/training_data/CT/test"),
        volume_idx, True
    )
    ct_img = resize256(extract_axial(ct_vol, axial_idx))

    # 3) load all CBCT & sCT
    cbct, sct = {}, {}
    for q in quals:
        cb_dir = os.path.expanduser(f"/Users/Niklas/thesis/training_data/CBCT/{q}/test")
        sc_dir = os.path.expanduser(
            f"/Users/Niklas/thesis/predictions/thesis-ready/{q}/best-model/50-steps-linear/volume-{volume_idx}"
        )
        v_cb = load_volume(cb_dir, volume_idx, True)
        v_sc = load_volume(sc_dir, volume_idx, False)
        cbct[q] = resize256(extract_axial(v_cb, axial_idx))
        sct[q]  = resize256(extract_axial(v_sc, axial_idx))

    # 4) make 5×4 grid: CT | _spacer_ | Q-label | CBCT | sCT
    nrows, ncols = len(quals), 5

    # width_ratios:    CT,   spacer,   Q,     CBCT,   sCT
    fig, axes = plt.subplots(
        nrows, 5,
        figsize=(5*2, nrows*2.5),
        gridspec_kw={'width_ratios': [2, 0.05, 0.0, 2, 2]},
        constrained_layout=False
    )

    # zero internal horizontal space:
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.5)

    for i, q in enumerate(quals):
        # grab each axis
        ax_ct    = axes[i, 0]
        ax_spacer= axes[i, 1]
        ax_label = axes[i, 2]
        ax_cb    = axes[i, 3]
        ax_sc    = axes[i, 4]

        # hide the spacer column completely
        ax_spacer.axis('off')

        # CT in col0 only if q==128
        if q == 128:
            ax_ct.imshow(ct_img, cmap="gray", vmin=-400, vmax=400)
            ax_ct.set_title("CT", pad=6)
        ax_ct.axis("off")

        # Q-label (tiny column)
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, f"           Q={q}",
                      va="center", ha="center",
                      fontsize=16)

        # CBCT and sCT
        ax_cb.imshow(cbct[q], cmap="gray", vmin=-400, vmax=400)
        ax_cb.axis("off")
        ax_sc.imshow(sct[q], cmap="gray", vmin=-400, vmax=400)
        ax_sc.axis("off")

        # column headers only on row0
        if i == 0:
            ax_cb.set_title("CBCT", pad=6)
            ax_sc.set_title("sCT (ours)", pad=6)

    out = "/Users/Niklas/thesis/figures/quality.pdf"
    fig.savefig(os.path.expanduser(out), bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()

if __name__=="__main__":
    plot_custom(volume_idx=8)
