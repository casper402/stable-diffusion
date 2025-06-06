#!/usr/bin/env python
import os
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ──────── constants ───────────────────────────────────────────────────────────
STEPS = 50

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

# ──────── slice-selection ─────────────────────────────────────────────────────
SLICE_SELECT = {
    3: None,
    8: (0, 354),
    # 8: (150, 150),
    12: (0, 320),
    26: None,
    32: (69, 269),
    33: (59, 249),
    35: (91, 268),
    # For volume 54, use inclusive range
    54: (0, 330),
    59: (0, 311),
    61: (0, 315),
    106: None,
    116: None,
    129: (5, 346)
}
VALID_VOLUMES = list(SLICE_SELECT.keys())

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
    return arr[TOP_CROP:RES_H - BOTTOM_CROP,
               LEFT_CROP:RES_W - RIGHT_CROP]

# ──────── metric functions ───────────────────────────────────────────────────
def compute_mae(a, b):
    return np.nanmean(np.abs(a - b))

def compute_rmse(a, b):
    return np.sqrt(np.nanmean((a - b) ** 2))

def compute_psnr(a, b, data_range):
    return psnr(b, a, data_range=data_range)

# ──────── file utils ──────────────────────────────────────────────────────────
def get_slice_files(folder, vol_idx, is_cbct=False):
    base = folder if is_cbct else os.path.join(folder, f"volume-{vol_idx}")
    pattern = os.path.join(base, f"volume-{vol_idx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    selector = SLICE_SELECT.get(vol_idx)
    if selector is None:
        return files
    # If explicit list, filter by those indices
    if isinstance(selector, list):
        valid = set(selector)
        files = [
            f for f in files
            if int(os.path.basename(f).split('_')[-1].split('.')[0]) in valid
        ]
    # If tuple, use inclusive range
    elif isinstance(selector, tuple) and len(selector) == 2:
        start, end = selector
        files = [
            f for f in files
            if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end
        ]
    return files

# ──────── single-slice comparison ─────────────────────────────────────────────
def compare_single(fname, test_folder, gt_folder, lm_folder, tm_folder, is_cbct):
    """
    Compare one slice (fname) between test, GT, liver-mask, tumor-mask.
    Returns a 12-tuple:
      (g_mae, g_rmse, g_psnr, g_ssim,
       l_mae, l_rmse, l_psnr, l_ssim,
       t_mae, t_rmse, t_psnr, t_ssim)
    """
    test = np.load(os.path.join(test_folder, fname))
    gt   = np.load(os.path.join(gt_folder,   fname))
    lm   = np.load(os.path.join(lm_folder,   fname)).astype(bool)
    tm   = np.load(os.path.join(tm_folder,   fname)).astype(bool)

    gt = crop_back(apply_transform(gt))
    lm = crop_back(apply_transform_to_mask(lm))
    tm = crop_back(apply_transform_to_mask(tm))

    if is_cbct:
        test = apply_transform(test)
    test = crop_back(test)

    # Global metrics
    g_mae  = compute_mae(test, gt)
    g_rmse = compute_rmse(test, gt)
    g_psnr = compute_psnr(test, gt, DATA_RANGE)
    g_ssim, ssim_map = ssim(gt, test, data_range=DATA_RANGE, full=True)

    # Helper to compute masked metrics
    def masked(fn, A, B, M):
        return fn(A[M], B[M]) if M.any() else np.nan

    # Liver-masked metrics
    l_mae  = masked(compute_mae,  test, gt, lm)
    l_rmse = masked(compute_rmse, test, gt, lm)
    l_psnr = masked(lambda x, y: compute_psnr(x, y, DATA_RANGE), test, gt, lm)
    l_ssim = np.nanmean(ssim_map[lm]) if lm.any() else np.nan

    # Tumor-masked metrics
    t_mae  = masked(compute_mae,  test, gt, tm)
    t_rmse = masked(compute_rmse, test, gt, tm)
    t_psnr = masked(lambda x, y: compute_psnr(x, y, DATA_RANGE), test, gt, tm)
    t_ssim = np.nanmean(ssim_map[tm]) if tm.any() else np.nan

    return (
        g_mae, g_rmse, g_psnr, g_ssim,
        l_mae, l_rmse, l_psnr, l_ssim,
        t_mae, t_rmse, t_psnr, t_ssim
    )

# ──────── batch & global eval ─────────────────────────────────────────────────
def compare_batch(vol_idx, test_base, gt_folder, lm_folder, tm_folder, is_cbct):
    """
    For a given volume index `vol_idx`, load all slice .npy files
    from `test_base/volume-{vol_idx}`, compare each slice to GT, liver mask, tumor mask,
    then average across slices to return a 12-tuple of
      (g_mae, g_rmse, g_psnr, g_ssim,
       l_mae, l_rmse, l_psnr, l_ssim,
       t_mae, t_rmse, t_psnr, t_ssim)
    """
    files = get_slice_files(test_base, vol_idx, is_cbct)
    if not files:
        print(f"No files for volume {vol_idx} in {test_base}")
        return None

    keys = [
        "g_mae","g_rmse","g_psnr","g_ssim",
        "l_mae","l_rmse","l_psnr","l_ssim",
        "t_mae","t_rmse","t_psnr","t_ssim"
    ]
    acc = {k: [] for k in keys}

    for path in files:
        fname = os.path.basename(path)
        vals = compare_single(
            fname,
            test_base if is_cbct else os.path.join(test_base, f"volume-{vol_idx}"),
            gt_folder,
            lm_folder,
            tm_folder,
            is_cbct
        )
        for k, v in zip(keys, vals):
            acc[k].append(v)

    # Return mean over all slices (ignore NaNs)
    return tuple(np.nanmean(acc[k]) for k in keys)

# ──────── evaluate multiple runs for volume 8 ────────────────────────────────
def evaluate_runs_for_volume_8(
    base_runs_folder,
    gt_folder,
    liver_mask_folder,
    tumor_mask_folder,
    n_runs=10,
    vol_idx=8,
    is_cbct=False
):
    """
    Assumes `base_runs_folder` contains subfolders "0", "1", ..., up to `n_runs-1`.
    Each subfolder must have a "volume-8" folder containing slice .npy files named
    "volume-8_slice_{XXX}.npy".  We call compare_batch() on each run to extract
    (g_mae, g_rmse, g_psnr, g_ssim) for volume 8, then print a table.
    """
    # Container for global metrics of shape (n_runs, 4)
    all_metrics = np.zeros((n_runs, 4), dtype=np.float32)

    for run_idx in range(n_runs):
        run_folder = os.path.join(base_runs_folder, str(run_idx))
        vals = compare_batch(
            vol_idx=vol_idx,
            test_base=run_folder,
            gt_folder=gt_folder,
            lm_folder=liver_mask_folder,
            tm_folder=tumor_mask_folder,
            is_cbct=is_cbct
        )
        if vals is None:
            all_metrics[run_idx, :] = np.nan
        else:
            # Take first four entries: (g_mae, g_rmse, g_psnr, g_ssim)
            all_metrics[run_idx, :] = np.array(vals[:4], dtype=np.float32)

    # Compute mean and standard deviation across runs
    means = np.nanmean(all_metrics, axis=0)
    stds  = np.nanstd(all_metrics, axis=0)

    # Print ASCII table
    header = "Run |     MAE |    RMSE |    PSNR |    SSIM "
    print(header)
    print("-" * len(header))
    for run_idx in range(n_runs):
        mae, rmse, psnr_val, ssim_val = all_metrics[run_idx]
        print(f"{run_idx:3d} | {mae:8.2f} | {rmse:7.2f} | {psnr_val:7.2f} | {ssim_val:7.3f}")
    print("-" * len(header))
    # Print “Mean ± SD”
    print(f"MEAN | "
          f"{means[0]:8.2f}±{stds[0]:.2f} | "
          f"{means[1]:7.2f}±{stds[1]:.2f} | "
          f"{means[2]:7.2f}±{stds[2]:.2f} | "
          f"{means[3]:7.3f}±{stds[3]:.3f}")

# ──────── main entrypoint ─────────────────────────────────────────────────────
def main():
    for _steps in [5, 10, 25, 50]:
        print(f"*** Evaluation for {_steps} steps ***")

        # Base folder that contains subfolders "0", "1", ..., "9"
        base_runs_folder = f"/Users/Niklas/thesis/predictions/thesis-ready/490/best-model/ddim/linear/{_steps}-steps"

        # Ground-truth CT slices
        gt_folder = os.path.expanduser("~/thesis/training_data/CT/test")
        # Liver masks
        liver_mask_folder = os.path.expanduser("~/thesis/training_data/liver/test")
        # Tumor masks
        tumor_mask_folder = os.path.expanduser("~/thesis/training_data/tumor/test")

        # Evaluate volume 8 for runs 0..9
        evaluate_runs_for_volume_8(
            base_runs_folder=base_runs_folder,
            gt_folder=gt_folder,
            liver_mask_folder=liver_mask_folder,
            tumor_mask_folder=tumor_mask_folder,
            n_runs=10,
            vol_idx=8,
            is_cbct=False
        )

        print("===========================================")

if __name__ == "__main__":
    main()
