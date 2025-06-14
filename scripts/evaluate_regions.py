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
DATA_RANGE = 2000.0    # CT full range: -1000 … 1000 HU
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))

# ──────── slice‐selection ─────────────────────────────────────────────────────
SLICE_SELECT = {
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

def apply_transform(img_np: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    out = gt_transform(t)
    return out.squeeze(0).numpy()

def crop_back(arr: np.ndarray) -> np.ndarray:
    return arr[TOP_CROP : RES_H - BOTTOM_CROP,
               LEFT_CROP : RES_W - RIGHT_CROP]

# ──────── intensity‐range definitions ──────────────────────────────────────────
HU_RANGES = [
    ("full",        -1000,  1000),
    ("neg1000_-150",-1000,  -150),
    ("neg150_150",  -150,    150),
    # ("neg100_100",  -100,    100),
    ("150_1000",     150,   1000),
]

# ──────── metric functions ───────────────────────────────────────────────────
def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    return np.nanmean(np.abs(a - b))

def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.nanmean((a - b) ** 2))

def compute_psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    return psnr(b, a, data_range=data_range)

# ──────── file utils ──────────────────────────────────────────────────────────
def get_slice_files(folder: str, vol_idx: int, is_cbct: bool=False) -> list:
    base = folder if is_cbct else os.path.join(folder, f"volume-{vol_idx}")
    pattern = os.path.join(base, f"volume-{vol_idx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    selector = SLICE_SELECT.get(vol_idx)
    if selector is None:
        return files
    if isinstance(selector, list):
        valid = set(selector)
        files = [f for f in files
                 if int(os.path.basename(f).split('_')[-1].split('.')[0]) in valid]
    elif isinstance(selector, tuple) and len(selector) == 2:
        start, end = selector
        files = [f for f in files
                 if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end]
    return files

# ──────── single‐slice, per‐range comparison ──────────────────────────────────
def compare_single_per_range(fname: str,
                             test_folder: str,
                             gt_folder: str,
                             is_cbct: bool=False) -> dict:
    test_path = os.path.join(test_folder, fname)
    gt_path   = os.path.join(gt_folder,   fname)
    test = np.load(test_path)
    gt   = np.load(gt_path)

    # Preprocess GT
    gt_pre = crop_back(apply_transform(gt))
    # Preprocess test
    if is_cbct:
        test_t = apply_transform(test)
    else:
        test_t = test.copy()
    test_pre = crop_back(test_t)

    full_ssim_value, full_ssim_map = ssim(gt_pre, test_pre,
                                          data_range=DATA_RANGE,
                                          full=True)

    results = {}
    for label, low, high in HU_RANGES:
        mask = (gt_pre >= low) & (gt_pre <= high)
        if not mask.any():
            results[label] = (np.nan, np.nan, np.nan, np.nan)
            continue

        gt_range   = gt_pre[mask]
        test_range = test_pre[mask]

        mae_val  = compute_mae(test_range, gt_range)
        rmse_val = compute_rmse(test_range, gt_range)
        psnr_val = compute_psnr(test_range, gt_range, DATA_RANGE)
        ssim_val = np.nanmean(full_ssim_map[mask])

        results[label] = (mae_val, rmse_val, psnr_val, ssim_val)

    return results

# ──────── batch & per‐volume evaluation ────────────────────────────────────────
def compare_batch_per_range(vol_idx: int,
                            test_base: str,
                            gt_folder: str,
                            is_cbct: bool=False) -> dict:
    files = get_slice_files(test_base, vol_idx, is_cbct)
    if not files:
        return {label: (np.nan, np.nan, np.nan, np.nan) for label,_,_ in HU_RANGES}

    acc = {
        label: { "mae": [], "rmse": [], "psnr": [], "ssim": [] }
        for label,_,_ in HU_RANGES
    }

    for path in files:
        fname = os.path.basename(path)
        per_slice = compare_single_per_range(
            fname,
            test_base if is_cbct else os.path.join(test_base, f"volume-{vol_idx}"),
            gt_folder,
            is_cbct
        )
        for label in per_slice:
            mae_val, rmse_val, psnr_val, ssim_val = per_slice[label]
            acc[label]["mae"].append(mae_val)
            acc[label]["rmse"].append(rmse_val)
            acc[label]["psnr"].append(psnr_val)
            acc[label]["ssim"].append(ssim_val)

    out = {}
    for label in acc:
        mae_arr  = np.array(acc[label]["mae"])
        rmse_arr = np.array(acc[label]["rmse"])
        psnr_arr = np.array(acc[label]["psnr"])
        ssim_arr = np.array(acc[label]["ssim"])

        mean_mae  = np.nanmean(mae_arr)
        mean_rmse = np.nanmean(rmse_arr)
        mean_psnr = np.nanmean(psnr_arr)
        mean_ssim = np.nanmean(ssim_arr)

        out[label] = (mean_mae, mean_rmse, mean_psnr, mean_ssim)

    return out

def run_eval_per_range(volumes: list,
                       test_base: str,
                       gt_folder: str,
                       is_cbct: bool=False) -> dict:
    results = {}
    for v in volumes:
        res = compare_batch_per_range(v, test_base, gt_folder, is_cbct)
        results[v] = res
    return results

# ──────── printing utility: one row per region ────────────────────────────────
def print_one_row_per_region(eval_sets: list, all_results: dict):
    """
    Print a table with one row per HU range. Columns are:
      {eval_label}_{MAE}, {eval_label}_{RMSE}, {eval_label}_{PSNR}, {eval_label}_{SSIM}
    """

    metrics = ["MAE", "RMSE", "PSNR", "SSIM"]

    # Build header: "Region" + for each eval_set: "label_MAE  label_RMSE  label_PSNR  label_SSIM"
    header_parts = ["Region".ljust(12)]
    for label, _, _ in eval_sets:
        for m in metrics:
            header_parts.append(f"{label+'_'+m:>12}")
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print("PER‐REGION AVERAGE METRICS (across all volumes)")
    print(header)
    print(sep)

    # For each region, compute average across volumes for each eval_set
    for range_label, _, _ in HU_RANGES:
        row_parts = [f"{range_label:12}"]
        for label, _, _ in eval_sets:
            # Gather this region's metrics over all volumes
            mae_vals  = []
            rmse_vals = []
            psnr_vals = []
            ssim_vals = []
            for v in all_results[label]:
                mae_val, rmse_val, psnr_val, ssim_val = all_results[label][v][range_label]
                mae_vals.append(mae_val)
                rmse_vals.append(rmse_val)
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)

            # Compute nan‐means
            mean_mae  = np.nanmean(np.array(mae_vals))
            mean_rmse = np.nanmean(np.array(rmse_vals))
            mean_psnr = np.nanmean(np.array(psnr_vals))
            mean_ssim = np.nanmean(np.array(ssim_vals))

            fmt_mae  = f"{mean_mae:12.2f}" if not np.isnan(mean_mae) else f"{'nan':>12}"
            fmt_rmse = f"{mean_rmse:12.2f}" if not np.isnan(mean_rmse) else f"{'nan':>12}"
            fmt_psnr = f"{mean_psnr:12.2f}" if not np.isnan(mean_psnr) else f"{'nan':>12}"
            fmt_ssim = f"{mean_ssim:12.3f}" if not np.isnan(mean_ssim) else f"{'nan':>12}"

            row_parts.extend([fmt_mae, fmt_rmse, fmt_psnr, fmt_ssim])

        print(" | ".join(row_parts))
    print(sep)

# ──────── main entrypoint ─────────────────────────────────────────────────────
def main():
    # Define your evaluation sets here; each is (label, folder, is_cbct)
    eval_sets = [
        # ("v7",  os.path.expanduser("~/thesis/predictions/predictions_controlnet_v7-data-augmentation"),                  False),
        # ("nl5",                  os.path.expanduser("~/thesis/predictions/predictions_tanh_v5"), False),
        ("nl2",                  os.path.expanduser("~/thesis/predictions/predictions_tanh_v2"), False),
        # ("nl6",                  os.path.expanduser("~/thesis/predictions/thesis-ready/490/best-model/50-steps-linear-tanh"), False),
    ]

    gt_folder = os.path.expanduser("~/thesis/training_data/CT/test")
    volumes = VALID_VOLUMES

    # Run evaluation for each set
    all_results = {}
    for label, folder, is_cbct in eval_sets:
        res = run_eval_per_range(volumes, folder, gt_folder, is_cbct)
        all_results[label] = res

    # Print one row per region (HU range)
    print_one_row_per_region(eval_sets, all_results)

if __name__ == "__main__":
    main()

