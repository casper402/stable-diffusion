import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ──────── constants ───────────────────────────────────────────────────────────
ORIG_H, ORIG_W    = 238, 366
PAD_L, PAD_T       = 0, 64
PAD_R, PAD_B       = 0, 64
RES_H, RES_W       = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))

gt_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W)),
])
mask_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=0),
    transforms.Resize((RES_H, RES_W), interpolation=InterpolationMode.NEAREST),
])

def transform_and_crop_ct(ct_np: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(ct_np).unsqueeze(0).float()
    padded = gt_transform(t).squeeze(0).numpy()
    return padded[TOP_CROP:RES_H - BOTTOM_CROP,
                  LEFT_CROP:RES_W - RIGHT_CROP]

def crop_pred(pred_np: np.ndarray) -> np.ndarray:
    """Just crop the already-transformed prediction back to the original field-of-view."""
    return pred_np[
        TOP_CROP:   RES_H - BOTTOM_CROP,
        LEFT_CROP:  RES_W - RIGHT_CROP
    ]

def transform_and_crop_mask(mask_np: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(mask_np.astype(np.uint8)).unsqueeze(0).float()
    padded = mask_transform(t).squeeze(0).numpy()
    binarized = padded > 0.5
    return binarized[TOP_CROP:RES_H - BOTTOM_CROP,
                     LEFT_CROP:RES_W - RIGHT_CROP]

def show_random_slice(volumes, ct_dir, mask1_dir, mask2_dir,
                      cbct_dir, pred_dir, max_slice,
                      vmin_ct=-1000, vmax_ct=1000, overlay_alpha=0.3):
    v = random.choice(volumes)            # draw from [3,8]
    z = random.randint(0, max_slice)
    z = 108
    pred_dir += f"/volume-{v}"
    fname = f"volume-{v}_slice_{z:03d}.npy"
    paths = [os.path.join(d, fname) for d in (ct_dir, mask1_dir, mask2_dir, cbct_dir, pred_dir)]
    if not all(os.path.isfile(p) for p in paths):
        for p in paths:
            print(f"Is {p} a file? {os.path.isfile(p)}")
        raise Exception("Something went wrong")

    print(f"Random pick → volume {v}, slice {z:03d}")

    # load raw
    ct   = np.load(os.path.join(ct_dir,    fname))
    lv   = np.load(os.path.join(mask1_dir, fname)).astype(bool)
    tu   = np.load(os.path.join(mask2_dir, fname)).astype(bool)
    cbct = np.load(os.path.join(cbct_dir,  fname))
    pred = np.load(os.path.join(pred_dir,  fname))

    # transform & crop
    ct   = transform_and_crop_ct(ct)
    cbct = transform_and_crop_ct(cbct)
    pred = crop_pred(pred)
    lv   = transform_and_crop_mask(lv)
    tu   = transform_and_crop_mask(tu)

    n_tumor = int(tu.sum())
    print(f"Tumor pixel count on this slice: {n_tumor}")

    # plot 6 panels in 2 rows × 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    ax0, ax1, ax2, ax3, ax4, ax5 = axes

    # 1) Liver mask
    ax0.imshow(lv, cmap="Reds", origin="lower", vmin=0, vmax=1,
               interpolation="none", aspect="equal")
    ax0.set_title(f"Liver Mask\n(v{v}, z={z:03d})")
    ax0.axis("off")

    # 2) Tumor mask
    ax1.imshow(tu, cmap="Blues", origin="lower", vmin=0, vmax=1,
               interpolation="none", aspect="equal")
    ax1.set_title(f"Tumor Mask (n={n_tumor})\n(v{v}, z={z:03d})")
    ax1.axis("off")

    # 3) CT only
    ax2.imshow(ct, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct,
               interpolation="none", aspect="equal")
    ax2.set_title(f"CT Only\n(v{v}, z={z:03d})")
    ax2.axis("off")

    # 4) CT + overlay
    ax3.imshow(ct, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct,
               interpolation="none", aspect="equal")
    ax3.imshow(np.ma.masked_where(~lv, lv), cmap="Reds", alpha=overlay_alpha,
               origin="lower", interpolation="none", aspect="equal")
    ax3.imshow(np.ma.masked_where(~tu, tu), cmap="Blues", alpha=overlay_alpha,
               origin="lower", interpolation="none", aspect="equal")
    ax3.set_title(f"CT + Overlay (n={n_tumor})")
    ax3.axis("off")

    # 5) CBCT only
    ax4.imshow(cbct, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct,
               interpolation="none", aspect="equal")
    ax4.set_title(f"CBCT Only\n(v{v}, z={z:03d})")
    ax4.axis("off")

    # 6) Prediction only
    ax5.imshow(pred, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct,
               interpolation="none", aspect="equal")
    ax5.set_title(f"Prediction Only\n(v{v}, z={z:03d})")
    ax5.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    # ─── CONFIG ─────────────────────────────────────────────────────
    volumes    = [8]
    ct_dir     = "/Users/Niklas/thesis/training_data/CT/test"
    mask1_dir  = "/Users/Niklas/thesis/training_data/liver/test"
    mask2_dir  = "/Users/Niklas/thesis/training_data/tumor/test"
    cbct_dir   = "/Users/Niklas/thesis/training_data/CBCT/test"
    pred_dir   = "/Users/Niklas/thesis/predictions/v1"

    max_slice  = 365
    vmin_ct, vmax_ct = -1000, 1000
    overlay_alpha   = 0.3
    # ──────────────────────────────────────────────────────────────────

    try:
        while True:
            show_random_slice(
                volumes,
                ct_dir, mask1_dir, mask2_dir,
                cbct_dir, pred_dir,
                max_slice,
                vmin_ct, vmax_ct, overlay_alpha
            )
    except KeyboardInterrupt:
        print("\nExiting viewer.")

if __name__ == "__main__":
    main()
