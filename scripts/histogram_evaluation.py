#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from torchvision.transforms import InterpolationMode

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

# ──────── slice-selection for evaluation ───────────────────────────────────────
SLICE_RANGES = {
    3: None, 8: (0, 354), 12: (0, 320), 26: None,
    32: (69, 269), 33: (59, 249), 35: (91, 268),
    54: (0, 330), 59: (0, 311), 61: (0, 315),
    106: None, 116: None, 129: (5, 346)
}
VALID_VOLUMES = list(SLICE_RANGES.keys())
VALID_VOLUMES = [8]

# ──────── transforms & crops ──────────────────────────────────────────────────
gt_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W)),
])
mask_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=0),
    transforms.Resize((RES_H, RES_W), interpolation=InterpolationMode.NEAREST),
])

def apply_transform(img_np):
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    return gt_transform(t).squeeze(0).numpy()

def apply_transform_to_mask(mask_np):
    t = torch.from_numpy(mask_np.astype(np.uint8)).unsqueeze(0).float()
    out = mask_transform(t).squeeze(0).numpy()
    return out > 0.5

def crop_back(arr):
    return arr[TOP_CROP:RES_H-BOTTOM_CROP, LEFT_CROP:RES_W-RIGHT_CROP]

# ──────── data loading utility ──────────────────────────────────────────────────
def get_slice_files(folder, vol_idx, is_cbct=False):
    """
    List slice file paths for a given volume index.
    If is_cbct is False, expects folder/<volume>-<idx>/volume-<idx>_slice_*.npy,
    otherwise folder/volume-<idx>_slice_*.npy.
    Applies SLICE_RANGES to filter slices.
    """
    base = folder if is_cbct else os.path.join(folder, f"volume-{vol_idx}")
    pattern = os.path.join(base, f"volume-{vol_idx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    rng = SLICE_RANGES.get(vol_idx)
    if rng:
        start, end = rng
        files = [f for f in files if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end]
    return files

# ──────── HISTOGRAM (Figure 3) ────────────────────────────────────────────────
def compute_hu_histogram(folder, vols, is_cbct=False, bins=np.linspace(-1000, 1000, 200)):
    hist = np.zeros(len(bins)-1)
    for v in vols:
        for fp in get_slice_files(folder, v, is_cbct):
            data = np.load(fp)
            if is_cbct:
                data = apply_transform(data)
            data = crop_back(data)
            hist += np.histogram(data.flatten(), bins=bins)[0]
    return hist


def plot_hu_distributions(gt_folder, cbct_folder, pred_folders, vols, save_path=None):
    """
    Plot HU histograms for GT (CT), CBCT, and multiple prediction methods.

    Parameters:
      gt_folder      : path to ground-truth CT slices
      cbct_folder    : path to raw CBCT slices
      pred_folders   : list of (label, folder_path) tuples for predictions
      vols           : list of volume indices to include
      save_path      : file path to save the figure (optional)
    """
    bins = np.linspace(-1000, 1000, 200)
    ctr = (bins[:-1] + bins[1:]) / 2

    # Compute histograms
    cth = compute_hu_histogram(gt_folder, vols, is_cbct=True, bins=bins)
    cbh = compute_hu_histogram(cbct_folder, vols, is_cbct=True, bins=bins)

    plt.figure(figsize=(8, 6))
    # Plot CT and CBCT
    plt.plot(ctr, cth/cth.sum() if cth.sum()>0 else cth, label='CT')
    plt.plot(ctr, cbh/cbh.sum() if cbh.sum()>0 else cbh, label='CBCT')

    # Plot each prediction
    for label, folder in pred_folders:
        pth = compute_hu_histogram(folder, vols, is_cbct=False, bins=bins)
        plt.plot(ctr, pth/pth.sum() if pth.sum()>0 else pth, label=label)

    plt.xlabel('Hounsfield Units (HU)')
    plt.ylabel('Normalized Frequency')
    plt.title('Figure 3: HU Distribution Comparison')
    plt.ylim(0, 0.15)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ──────── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vols = VALID_VOLUMES

    gt_folder         = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder       = os.path.expanduser("~/thesis/training_data/CBCT/test")

    # List of (label, prediction_folder) for multiple methods
    pred_folders = [
        # ("v3", os.path.expanduser("~/thesis/predictions/predictions_controlnet_v3")),
        ("v7", os.path.expanduser("~/thesis/predictions/predictions_controlnet_v7-data-augmentation")),
        ("perceptual", os.path.expanduser("~/thesis/predictions/predctions_controlnet_from_unet_trained_after_joint_v2")),
    ]

    plot_hu_distributions(gt_folder, cbct_folder, pred_folders, vols)
