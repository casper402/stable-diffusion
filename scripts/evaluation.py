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
# specify volumes and optional slice ranges (None => all slices)
SLICE_RANGES = {
    3: None,           # use all slices
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

# ──────── transforms & crops ─────────────────────────────────────────────────
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
    return arr[
        TOP_CROP:   RES_H - BOTTOM_CROP,
        LEFT_CROP:  RES_W - RIGHT_CROP
    ]

# ──────── metric functions ───────────────────────────────────────────────────
def compute_mae(a, b):
    return np.nanmean(np.abs(a - b))  # ignore NaNs

def compute_rmse(a, b):
    return np.sqrt(np.nanmean((a - b)**2))  # ignore NaNs

def compute_psnr(a, b, data_range):
    return psnr(b, a, data_range=data_range)  # skimage PSNR skips NaNs

# ──────── per‐slice comparison ────────────────────────────────────────────────
def compare_single(fname, test_folder, gt_folder, lm_folder, tm_folder, is_cbct):
    test = np.load(os.path.join(test_folder, fname))
    gt   = np.load(os.path.join(gt_folder,   fname))
    lm   = np.load(os.path.join(lm_folder,   fname)).astype(bool)
    tm   = np.load(os.path.join(tm_folder,   fname)).astype(bool)

    gt   = crop_back(apply_transform(gt))
    lm   = crop_back(apply_transform_to_mask(lm))
    tm   = crop_back(apply_transform_to_mask(tm))
    if is_cbct:
        test = apply_transform(test)
    test = crop_back(test)

    g_mae  = compute_mae(test, gt)
    g_rmse = compute_rmse(test, gt)
    g_psnr = compute_psnr(test, gt, DATA_RANGE)
    g_ssim, ssim_map = ssim(gt, test, data_range=DATA_RANGE, full=True)

    def masked(fn, A, B, M):
        return fn(A[M], B[M]) if M.any() else np.nan

    l_mae  = masked(compute_mae,  test, gt, lm)
    l_rmse = masked(compute_rmse, test, gt, lm)
    l_psnr = masked(lambda x, y: compute_psnr(x, y, DATA_RANGE), test, gt, lm)
    l_ssim = np.nanmean(ssim_map[lm]) if lm.any() else np.nan

    t_mae  = masked(compute_mae,  test, gt, tm)
    t_rmse = masked(compute_rmse, test, gt, tm)
    t_psnr = masked(lambda x, y: compute_psnr(x, y, DATA_RANGE), test, gt, tm)
    t_ssim = np.nanmean(ssim_map[tm]) if tm.any() else np.nan

    return (
        g_mae, g_rmse, g_psnr, g_ssim,
        l_mae, l_rmse, l_psnr, l_ssim,
        t_mae, t_rmse, t_psnr, t_ssim
    )

def compare_batch(vol_idx, test_base, gt_folder, lm_folder, tm_folder, is_cbct):
    test_folder = test_base if is_cbct else os.path.join(test_base, f"volume-{vol_idx}")
    pattern = os.path.join(test_folder, f"volume-{vol_idx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for volume {vol_idx}")
        return None

    # Apply slice-range filter if defined
    rng = SLICE_RANGES.get(vol_idx)
    if rng is not None:
        start, end = rng
        files = [f for f in files
                 if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end]
        if not files:
            print(f"No files in range {start}-{end} for volume {vol_idx}")
            return None

    keys = ["g_mae","g_rmse","g_psnr","g_ssim",
            "l_mae","l_rmse","l_psnr","l_ssim",
            "t_mae","t_rmse","t_psnr","t_ssim"]
    acc = {k: [] for k in keys}
    for p in files:
        vals = compare_single(os.path.basename(p),
                              test_folder, gt_folder,
                              lm_folder, tm_folder, is_cbct)
        for k, v in zip(keys, vals):
            acc[k].append(v)
    return tuple(np.nanmean(acc[k]) for k in keys)

def run_eval(vols, base_folder, is_cbct):
    res = {}
    for v in vols:
        vals = compare_batch(v, base_folder,
                             gt_folder,
                             liver_mask_folder,
                             tumor_mask_folder,
                             is_cbct)
        if vals is not None:
            res[v] = vals
    return res

if __name__ == "__main__":
    volumes = [3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129]

    cbct_base    = os.path.expanduser("~/thesis/training_data/CBCT/test")
    pred_base    = os.path.expanduser("~/thesis/predictions/v1")

    gt_folder          = os.path.expanduser("~/thesis/training_data/CT/test")
    liver_mask_folder  = os.path.expanduser("~/thesis/training_data/liver/test")
    tumor_mask_folder  = os.path.expanduser("~/thesis/training_data/tumor/test")

    eval_sets = [
        ("CBCT", cbct_base, True),
        ("Pred", pred_base, False),
    ]

    results = {label: run_eval(volumes, folder, is_cbct)
               for label, folder, is_cbct in eval_sets}

    # ──────── 1) GLOBAL METRICS TABLE ────────────────────────────────────────
    metrics = ["MAE","RMSE","PSNR","SSIM"]

    # header
    hdr_g = "Vol".rjust(4) + " | " + " | ".join(
        " ".join(f"{(label+'_'+m):>10}" for m in metrics)
        for label,_,_ in eval_sets
    )
    print("\nGLOBAL METRICS")
    print(hdr_g)
    print("-" * len(hdr_g))

    # per-volume rows
    for v in volumes:
        parts = []
        for label,_,_ in eval_sets:
            vals = results[label].get(v, [np.nan]*len(metrics))
            part = " ".join(
                f"{vals[k]:>10.2f}" if m != "SSIM" else f"{vals[k]:>10.3f}"
                for k, m in enumerate(metrics)
            )
            parts.append(part)
        print(f"{v:4d} | " + " | ".join(parts))

    # overall
    def overall(rs):
        arr = np.stack(list(rs.values()), axis=0)
        return np.nanmean(arr, axis=0)

    overall_results = {label: overall(results[label]) for label,_,_ in eval_sets}

    print("-" * len(hdr_g))
    overall_parts = []
    for label,_,_ in eval_sets:
        vals = overall_results[label]
        part = " ".join(
            f"{vals[k]:>10.2f}" if m != "SSIM" else f"{vals[k]:>10.3f}"
            for k, m in enumerate(metrics)
        )
        overall_parts.append(part)
    print(f"{'ALL':>4} | " + " | ".join(overall_parts))

    # ──────── 2) LIVER & TUMOR METRICS TABLE ────────────────────────────────
    hdr_lt = (
        "Vol".rjust(4) + " | " +
        "Region".rjust(6) + " | " +
        " | ".join(
            " ".join(f"{(label+'_'+m):>10}" for m in metrics)
            for label,_,_ in eval_sets
        )
    )
    print("\nLIVER & TUMOR METRICS")
    print(hdr_lt)
    print("-" * len(hdr_lt))

    for v in volumes:
        for region, base_off in [("Liver", 4), ("Tumor", 8)]:
            row_parts = []
            for label,_,_ in eval_sets:
                vals = results[label].get(v, [np.nan]*12)
                part = " ".join(
                    f"{vals[base_off + k]:>10.2f}" if m != "SSIM" else f"{vals[base_off + k]:>10.3f}"
                    for k, m in enumerate(metrics)
                )
                row_parts.append(part)
            print(f"{v:4d} | {region:>6} | " + " | ".join(row_parts))

    print("-" * len(hdr_lt))
    for region, base_off in [("Liver", 4), ("Tumor", 8)]:
        row_parts = []
        for label,_,_ in eval_sets:
            vals = overall_results[label]
            part = " ".join(
                f"{vals[base_off + k]:>10.2f}" if m != "SSIM" else f"{vals[base_off + k]:>10.3f}"
                for k, m in enumerate(metrics)
            )
            row_parts.append(part)
        print(f"{'ALL':>4} | {region:>6} | " + " | ".join(row_parts))
