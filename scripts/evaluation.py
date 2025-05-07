#!/usr/bin/env python
import os
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from collections import namedtuple

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
    3: None,
    8: (0, 354),
    12: (0, 320),
    26: None,
    32: (69, 269), # confirmed
    33: (59, 249),
    35: (91, 268),
    54: (0, 330),
    59: (0, 311),
    61: (0, 315),
    106: None,
    116: None,
    129: (5, 346)
}

# all volumes are valid; None means include all slices
VALID_VOLUMES = list(SLICE_RANGES.keys())

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
    return arr[
        TOP_CROP:   RES_H - BOTTOM_CROP,
        LEFT_CROP:  RES_W - RIGHT_CROP
    ]

# ──────── metric functions ───────────────────────────────────────────────────
def compute_mae(a, b):
    return np.nanmean(np.abs(a - b))

def compute_rmse(a, b):
    return np.sqrt(np.nanmean((a - b)**2))

def compute_psnr(a, b, data_range):
    return psnr(b, a, data_range=data_range)

# ──────── per-slice comparison ────────────────────────────────────────────────
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

    g_mae, g_rmse, g_psnr = (
        compute_mae(test, gt),
        compute_rmse(test, gt),
        compute_psnr(test, gt, DATA_RANGE)
    )
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

# ──────── helper to list & filter slice files ─────────────────────────────────
def get_slice_files(folder, vol_idx, is_cbct=False):
    base = folder if is_cbct else os.path.join(folder, f"volume-{vol_idx}")
    pattern = os.path.join(base, f"volume-{vol_idx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    rng = SLICE_RANGES.get(vol_idx)
    if rng is not None:
        start, end = rng
        files = [
            f for f in files
            if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end
        ]
    return files

# ──────── batch & global eval ─────────────────────────────────────────────────
def compare_batch(vol_idx, test_base, gt_folder, lm_folder, tm_folder, is_cbct):
    files = get_slice_files(test_base, vol_idx, is_cbct)
    if not files:
        print(f"No files for volume {vol_idx}")
        return None

    keys = ["g_mae","g_rmse","g_psnr","g_ssim",
            "l_mae","l_rmse","l_psnr","l_ssim",
            "t_mae","t_rmse","t_psnr","t_ssim"]
    acc = {k: [] for k in keys}
    for p in files:
        vals = compare_single(os.path.basename(p),
                              test_base if is_cbct else os.path.join(test_base, f"volume-{vol_idx}"),
                              gt_folder, lm_folder, tm_folder, is_cbct)
        for k, v in zip(keys, vals):
            acc[k].append(v)
    return tuple(np.nanmean(acc[k]) for k in keys)

def run_eval(vols, base_folder, is_cbct, gt_folder, lm_folder, tm_folder):
    res = {}
    for v in vols:
        vals = compare_batch(v, base_folder,
                             gt_folder, lm_folder, tm_folder, is_cbct)
        if vals is not None:
            res[v] = vals
    return res

# ──────── per-slice stats collector with range filtering ──────────────────────
SliceStat = namedtuple("SliceStat", [
    "volume", "slice_idx",
    "cbct_mae", "pred_mae", "delta_mae",
    "cbct_ssim", "pred_ssim", "delta_ssim",
    "tumor_pixels", "t_pred_mae"
])

def collect_slice_stats(volumes, cbct_base, pred_base,
                        gt_folder, lm_folder, tm_folder):
    stats = []
    for v in volumes:
        files = get_slice_files(cbct_base, v, is_cbct=True)
        for cbct_f in files:
            fname = os.path.basename(cbct_f)
            idx = int(fname.split('_')[-1].split('.')[0])
            cbct_vals = compare_single(
                fname, cbct_base,
                gt_folder, lm_folder, tm_folder,
                is_cbct=True
            )
            pred_folder = os.path.join(pred_base, f"volume-{v}")
            pred_vals = compare_single(
                fname, pred_folder,
                gt_folder, lm_folder, tm_folder,
                is_cbct=False
            )
            tm_raw = np.load(os.path.join(tm_folder, fname)).astype(bool)
            tm_mask = crop_back(apply_transform_to_mask(tm_raw))
            tumor_pixels = int(tm_mask.sum())
            cbct_mae   = cbct_vals[0]
            pred_mae   = pred_vals[0]
            delta_mae  = abs(cbct_mae - pred_mae)
            cbct_ssim  = cbct_vals[3]
            pred_ssim  = pred_vals[3]
            delta_ssim = abs(cbct_ssim - pred_ssim)
            t_pred_mae = pred_vals[8]
            stats.append(
                SliceStat(
                    volume=v,
                    slice_idx=idx,
                    cbct_mae=cbct_mae,
                    pred_mae=pred_mae,
                    delta_mae=delta_mae,
                    cbct_ssim=cbct_ssim,
                    pred_ssim=pred_ssim,
                    delta_ssim=delta_ssim,
                    tumor_pixels=tumor_pixels,
                    t_pred_mae=t_pred_mae
                )
            )
    return stats

if __name__ == "__main__":
    volumes = VALID_VOLUMES

    cbct_base           = os.path.expanduser("~/thesis/training_data/CBCT/test")
    cbct490_base        = os.path.expanduser("~/thesis/training_data/CBCT/scaled-490")

    # v1 
    pred_base           = os.path.expanduser("~/thesis/predictions/v1")
    predspeed_base      = os.path.expanduser("~/thesis/predictions/v1_speed")
    pred490_base        = os.path.expanduser("~/thesis/predictions/v1_490")
    pred490speed_base   = os.path.expanduser("~/thesis/predictions/v1_490_speed")

    # v2
    v2_pred490speed_base   = os.path.expanduser("~/thesis/predictions/v2_490_speed")
    v2_pred490speed100steps_base   = os.path.expanduser("~/thesis/predictions/v2_490_speed_100steps")
    v2_pred490speedstepsize20_base   = os.path.expanduser("~/thesis/predictions/v2_490_speed_stepsize20")
    v2_pred490speedstepsize20v2_base   = os.path.expanduser("~/thesis/predictions/v2_490_speed_stepsize20_v2")
    v2_cbct   = os.path.expanduser("~/thesis/predictions/predictionsV2-490-50steps_v2_cbct")
    

    gt_folder           = os.path.expanduser("~/thesis/training_data/CT/test")
    liver_mask_folder   = os.path.expanduser("~/thesis/training_data/liver/test")
    tumor_mask_folder   = os.path.expanduser("~/thesis/training_data/tumor/test")

    eval_sets = [
        ("noise",  v2_pred490speedstepsize20v2_base, False),
        ("with cbct",  v2_cbct, False),
    ]

    # ──────── 1) GLOBAL & REGION‑BASED EVAL ─────────────────────────────────
    results = {
        label: run_eval(volumes, folder, is_cbct,
                        gt_folder, liver_mask_folder, tumor_mask_folder)
        for label, folder, is_cbct in eval_sets
    }

    metrics = ["MAE","RMSE","PSNR","SSIM"]

    # -- global table --
    hdr_g = "Vol".rjust(4) + " | " + " | ".join(
        " ".join(f"{(label+'_'+m):>10}" for m in metrics)
        for label,_,_ in eval_sets
    )
    print("\nGLOBAL METRICS")
    print(hdr_g)
    print("-" * len(hdr_g))
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
    def overall(rr):
        arr = np.stack(list(rr.values()), axis=0)
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

    # -- liver & tumor table --
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

    raise Exception("stopping early")

    # ──────── 2) PER-SLICE STATISTICS ───────────────────────────────────────────
    slice_stats = collect_slice_stats(
        volumes,
        cbct_base,
        pred_base,
        gt_folder,
        liver_mask_folder,
        tumor_mask_folder
    )

    # (a) Top-5 ΔMAE per volume
    per_vol_max_mae = {}
    for s in slice_stats:
        cur = per_vol_max_mae.get(s.volume)
        if cur is None or s.delta_mae > cur.delta_mae:
            per_vol_max_mae[s.volume] = s
    top5_delta_mae = sorted(per_vol_max_mae.values(),
                             key=lambda x: x.delta_mae,
                             reverse=True)[:5]

    # (b) Top-5 ΔSSIM per volume
    per_vol_max_ssim = {}
    for s in slice_stats:
        cur = per_vol_max_ssim.get(s.volume)
        if cur is None or s.delta_ssim > cur.delta_ssim:
            per_vol_max_ssim[s.volume] = s
    top5_delta_ssim = sorted(per_vol_max_ssim.values(),
                              key=lambda x: x.delta_ssim,
                              reverse=True)[:5]

    # (c) Top-5 worst Pred MAE
    top5_pred_mae = sorted(slice_stats,
                            key=lambda x: x.pred_mae,
                            reverse=True)[:5]

    # (d) Top-5 worst Pred SSIM
    top5_pred_ssim = sorted(slice_stats,
                             key=lambda x: x.pred_ssim)[:5]

    # (e) Top-5 best Pred tumor MAE (only slices with >300 tumor px)
    filtered = [s for s in slice_stats if s.tumor_pixels > 300]
    top5_tumor_best = sorted(filtered, key=lambda x: x.t_pred_mae)[:5]

    # helper to print tables
    def print_slice_table(title, stats, fields):
        print(f"\n{title}")
        hdr = "Vol Slice | " + " | ".join(f"{f:>10}" for f in fields)
        print(hdr)
        print("-" * len(hdr))
        for s in stats:
            vals = [getattr(s, f) for f in fields]
            print(f"{s.volume:>3d} {s.slice_idx:>5d} | " +
                  " | ".join(f"{v:10.3f}" for v in vals))

    print_slice_table(
        "Top 5 ΔMAE (one slice per volume):",
        top5_delta_mae,
        ["cbct_mae", "pred_mae", "delta_mae"]
    )
    print_slice_table(
        "Top 5 ΔSSIM (one slice per volume):",
        top5_delta_ssim,
        ["cbct_ssim", "pred_ssim", "delta_ssim"]
    )
    print_slice_table(
        "Top 5 worst Pred MAE (all volumes):",
        top5_pred_mae,
        ["pred_mae", "cbct_mae"]
    )
    print_slice_table(
        "Top 5 worst Pred SSIM (all volumes):",
        top5_pred_ssim,
        ["pred_ssim", "cbct_ssim"]
    )

    print("\nTop 5 best Pred tumor MAE (≥300 tumor px):")
    hdr_t = "Vol Slice | Tumor_px | Pred_tumor_MAE"
    print(hdr_t)
    print("-" * len(hdr_t))
    for s in top5_tumor_best:
        print(f"{s.volume:>3d} {s.slice_idx:>5d} | "
              f"{s.tumor_pixels:>8d} | {s.t_pred_mae:>16.3f}")
