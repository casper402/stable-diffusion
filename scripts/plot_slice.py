#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from evaluation import compute_mae, compute_rmse, compute_psnr, DATA_RANGE, ssim
from PIL import Image
from torchvision import transforms
import random

SLICE_RANGES = {
    3: None,
    8: (0, 354),
    12: (0, 320),
    26: None,
    32: (69, 269),
    33: (59, 249),
    35: (91, 268),
    54: (0, 330),
    59: (0, 311),
    61: (0, 315),
    106: None,
    116: None,
    129: (5, 346)
}

# ──────── constants ───────────────────────────────────────────────────────────
DATA_RANGE = 2000.0    # CT range -1000…1000
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))

transform = transforms.Compose([
    transforms.Pad((0, 64, 0, 64), fill=-1000),
    transforms.Resize((256, 256)),
])

def apply_transform(np_img):
    """Convert NumPy→PIL→transformed PIL→NumPy"""
    pil = Image.fromarray(np_img)
    out = transform(pil)
    return np.array(out)

def crop_back(arr):
    return arr[
        TOP_CROP:   RES_H - BOTTOM_CROP,
        LEFT_CROP:  RES_W - RIGHT_CROP
    ]

def plot_multi_side_by_side(test_dirs, gt_dir, volume_idx, slice_num):
    """
    Plots the ground truth image and an arbitrary number of test images in a grid
    with up to 4 plots per row. Also computes MAE, RMSE, PSNR, and SSIM for each
    test image relative to the ground truth and adds these values to each test
    image's subplot title.

    As you move the mouse over the images, an annotation updates to show the pixel value
    at the cursor for the ground truth and, for each test image, its pixel value along with 
    the difference (Δ) from the ground truth.
    """
    # Construct filename
    base_name = f"volume-{volume_idx}"
    filename = f"{base_name}_slice_{slice_num:03d}.npy"

    # Load ground truth
    gt_file = os.path.join(gt_dir, filename)
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    gt_image = np.load(os.path.join(gt_dir, filename))
    gt_image = apply_transform(gt_image)
    gt_image = crop_back(gt_image)

    test_images = []
    test_names  = []
    for idx, test_dir in enumerate(test_dirs):
        img = np.load(os.path.join(test_dir, filename))

        cbct_names = ["test", "scaled-490"]
        if os.path.basename(test_dir) in cbct_names:
            img = apply_transform(img)

        img = crop_back(img)

        test_images.append(img)
        test_names.append(os.path.basename(os.path.normpath(test_dir)))

    total_plots = 1 + len(test_images)
    ncols = 4 if total_plots >= 4 else total_plots
    nrows = (total_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 6 * nrows))
    axes = axes.flatten()
    for j in range(total_plots, len(axes)):
        axes[j].axis("off")

    # Plot GT
    axes[0].imshow(gt_image, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Plot tests with metrics
    for i, test_img in enumerate(test_images, start=1):
        mae_val  = compute_mae(test_img, gt_image)
        rmse_val = compute_rmse(test_img, gt_image)
        psnr_val = compute_psnr(test_img, gt_image, data_range=DATA_RANGE)
        ssim_val = ssim(gt_image, test_img, data_range=DATA_RANGE)
        title = (
            f"MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}\n"
            f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.2f}"
        )
        axes[i].imshow(test_img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    info_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=10)

    def on_move(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            info_text.set_text("")
            fig.canvas.draw_idle()
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        h, w = gt_image.shape
        if not (0 <= x < w and 0 <= y < h):
            info_text.set_text("")
            fig.canvas.draw_idle()
            return

        gt_val = gt_image[y, x]
        parts = [f"({x}, {y}) | GT: {gt_val:.2f}"]
        for idx, test_img in enumerate(test_images):
            tv = test_img[y, x]
            diff = tv - gt_val
            parts.append(f"{test_names[idx]}: {tv:.2f} (Δ={diff:+.2f})")
        info_text.set_text(" | ".join(parts))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.suptitle(f"Volume {volume_idx} - Slice {slice_num}", fontsize=16)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()

def plot(volume_idx, slice_num):
    test_dirs = [
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/v1/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/v1_speed/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/basic/volume-{volume_idx}"),
        # os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/test"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/v1_490/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/v1_490_speed/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/v2_490_speed/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/v2_490_speed_stepsize20/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/v2_490_speed_stepsize20_v2/volume-{volume_idx}"),
        os.path.expanduser(f"/Users/Niklas/thesis/predictions/predctions_controlnet_v3/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/predictions-v3-stepsize1/volume-{volume_idx}"),
        # os.path.expanduser(f"/Users/Niklas/thesis/predictions/predictions_after_joint_round2/volume-{volume_idx}"),
        os.path.expanduser(f"/Users/Niklas/thesis/predictions/predictions_controlnet_v7-data-augmentation/volume-{volume_idx}"),
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/scaled-490"),
    ]
    gt_dir = os.path.expanduser("/Users/Niklas/thesis/training_data/CT/test")
    
    plot_multi_side_by_side(test_dirs, gt_dir, volume_idx, slice_num)

def plot_random_slice(volume_idx):
    lb, ub = 0, 363
    if volume_idx in SLICE_RANGES and SLICE_RANGES[volume_idx] is not None:
        lb, ub = SLICE_RANGES[volume_idx]
    while True:
        slice_num = random.randint(lb, ub)
        plot(volume_idx, slice_num)


def plot_specific():
    volume_idx = 54
    slice_num = 266
    plot(volume_idx, slice_num)

if __name__ == "__main__":
    plot_specific()
    # plot_random_slice(33)
