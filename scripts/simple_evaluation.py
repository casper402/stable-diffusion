#!/usr/bin/env python3
import os
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

# -----------------------------------------
# Configuration: set your folders and volume index here
CBCT_FOLDER = os.path.expanduser("~/thesis/training_data/CBCT/490/test")
GT_FOLDER   = os.path.expanduser("~/thesis/training_data/CT/test")
VOLUME_IDX  = 54  # e.g., process volume-32
# -----------------------------------------

data_range = 2000.0  # CT intensity range (-1000 to 1000)

def compute_mae(a, b):
    return np.nanmean(np.abs(a - b))

def compute_rmse(a, b):
    return math.sqrt(np.nanmean((a - b) ** 2))

def evaluate_volume(cbct_folder, gt_folder, vol_idx):
    """
    Evaluate per-slice MAE, RMSE, SSIM for a single CBCT volume against GT CT.
    Only processes files matching 'volume-{vol_idx}_slice_*.npy'.
    """
    pattern = os.path.join(
        cbct_folder,
        f"volume-{vol_idx}_slice_*.npy"
    )
    # Gather and sort slice files by numeric slice index
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
    )

    if not files:
        print(f"No files found for volume {vol_idx} in {cbct_folder}")
        return

    maes, rmses, ssims = [], [], []

    for path in files:
        fname = os.path.basename(path)
        idx = int(fname.split('_')[-1].split('.')[0])

        cbct = np.load(path)
        gt   = np.load(os.path.join(gt_folder, fname))

        mae_val  = compute_mae(cbct, gt)
        rmse_val = compute_rmse(cbct, gt)
        ssim_val = ssim(gt, cbct, data_range=data_range)

        maes.append(mae_val)
        rmses.append(rmse_val)
        ssims.append(ssim_val)

        print(f"Slice {idx:03d}: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, SSIM={ssim_val:.4f}")

    # Print overall means
    print()
    print(f"Mean   : MAE={np.mean(maes):.4f}, RMSE={np.mean(rmses):.4f}, SSIM={np.mean(ssims):.4f}")


if __name__ == "__main__":
    evaluate_volume(CBCT_FOLDER, GT_FOLDER, VOLUME_IDX)
