import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
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

# interpolation variants and step sizes
target_quality = 490
variants = ['Linear', 'Power']
steps = [1, 2, 5, 10, 25, 1000]

# ───── transformation helpers ─────────────────────────────────────────────────
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

def extract_axial(vol, idx):
    return np.fliplr(vol[idx])

def extract_coronal(vol, idx):
    return np.fliplr(np.flipud(vol[:, idx, :]))

def resize256(slice2d):
    return np.array(Image.fromarray(slice2d.astype(np.int16))
                    .resize((256,256), Image.BILINEAR))

# ───── plotting function for variants & steps ─────────────────────────────────
def plot_variants(volume_idx):
    # style
    plt.rcParams["font.family"]    = "serif"
    plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
    plt.rcParams["font.size"]      = 16
    plt.rcParams["axes.titlesize"] = 16

    # determine slice ranges from a sample
    sample_dir = os.path.expanduser(f"/Users/Niklas/thesis/training_data/CBCT/{target_quality}/test")
    sample = load_volume(sample_dir, volume_idx, True)
    Z, H, W = sample.shape

    if volume_idx in SLICE_RANGES and SLICE_RANGES[volume_idx]:
        lb, ub = SLICE_RANGES[volume_idx]
        ub = min(ub, Z-1)
    else:
        lb, ub = 0, Z-1
    axial_idx = 150 if lb <= 150 <= ub else random.randint(lb, ub)
    print(f"Axial idx = {axial_idx} ({lb}–{ub})")

    coronal_idx = 120
    print(f"Coronal idx = {coronal_idx}")

    # load all variants & steps
    sct_ax = {var: {} for var in variants}
    sct_cor = {var: {} for var in variants}
    for var in variants:
        for s in steps:
            if var == 'Power' and s in (1, 1000):
                continue
            sc_dir = os.path.expanduser(
                f"/Users/Niklas/thesis/predictions/thesis-ready/{target_quality}/best-model/ddim/{var}/{s}-steps/0/volume-{volume_idx}"
            )
            v_sc = load_volume(sc_dir, volume_idx, False)
            sct_ax[var][s] = resize256(extract_axial(v_sc, axial_idx))
            sct_cor[var][s] = resize256(extract_coronal(v_sc, coronal_idx))

    # ─── generate a 256×256 placeholder with centered text ───────────────────────
    ph_size = (256, 256)
    bg_color = 200        # light gray
    text_color = 80       # darker gray
    placeholder_img = Image.new('L', ph_size, color=bg_color)
    draw = ImageDraw.Draw(placeholder_img)
    placeholder_text = ""
    font = ImageFont.load_default()

    # manually measure multiline text
    lines = placeholder_text.split('\n')
    spacing = 4
    widths, heights = [], []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w_line = bbox[2] - bbox[0]
        h_line = bbox[3] - bbox[1]
        widths.append(w_line)
        heights.append(h_line)
    text_w = max(widths)
    text_h = sum(heights) + spacing * (len(lines) - 1)

    x = (ph_size[0] - text_w) / 2
    y = (ph_size[1] - text_h) / 2

    # draw each line
    y_offset = y
    for line, h_line in zip(lines, heights):
        draw.text((x, y_offset), line, fill=text_color, font=font, align='center')
        y_offset += h_line + spacing

    placeholder = np.array(placeholder_img)

    # plot grid: step label | [variant_ax | variant_cor] per variant
    nrows = len(steps)
    ncols = 1 + 2 * len(variants)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2, nrows * 2.5),
        gridspec_kw={'width_ratios': [0.2] + [2] * (ncols - 1)},
        constrained_layout=True
    )

    for i, s in enumerate(steps):
        # label column
        ax_label = axes[i, 0]
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5, f"Steps={s}",
            va="center", ha="center", rotation="vertical",
            transform=ax_label.transAxes, fontsize=16
        )

        imgs = []
        for var in variants:
            if var == 'Power' and s in (1, 1000):
                # insert placeholder for both axial & coronal slots
                imgs.append((placeholder, f"sCT Axial ({var})"))
                imgs.append((placeholder, f"sCT Coronal ({var})"))
            else:
                imgs.append((sct_ax[var][s], f"sCT Axial ({var})"))
                imgs.append((sct_cor[var][s], f"sCT Coronal ({var})"))

        # plot row
        for j, (img, title) in enumerate(imgs, start=1):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=-400, vmax=400)
            ax.axis("off")
            if i == 0 and title:
                ax.set_title(title, pad=6)

    out_path = "/Users/Niklas/thesis/figures/ax_cor_variants.pdf"
    fig.savefig(os.path.expanduser(out_path), bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_variants(volume_idx=8)
