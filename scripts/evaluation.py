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

# ──────── slice-selection ─────────────────────────────────────────────────────
SLICE_SELECT = {
    3: None,
    8: (0, 354),
    12: (0, 320),
    26: None,
    32: (69, 269),
    33: (59, 249),
    35: (91, 268),
    # For volume 54, use explicit list instead of a range
    # 54: [0, 4, 11, 19, 26, 33, 40, 48, 55, 62, 70, 77, 84, 91, 99, 106, 113, 120, 128, 135, 142, 149, 157, 164, 171, 179, 186, 193, 200, 208, 215, 222, 229, 237, 244, 251, 259, 266, 273, 280, 2888, 295, 317, 324],
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
def compute_mae(a, b): return np.nanmean(np.abs(a - b))
def compute_rmse(a, b): return np.sqrt(np.nanmean((a - b)**2))
def compute_psnr(a, b, data_range): return psnr(b, a, data_range=data_range)

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
        files = [f for f in files
                 if int(os.path.basename(f).split('_')[-1].split('.')[0]) in valid]
    # If tuple, use inclusive range
    elif isinstance(selector, tuple) and len(selector) == 2:
        start, end = selector
        files = [f for f in files
                 if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end]
    return files

# ──────── single-slice comparison ─────────────────────────────────────────────
def compare_single(fname, test_folder, gt_folder, lm_folder, tm_folder, is_cbct):
    test = np.load(os.path.join(test_folder, fname))
    gt   = np.load(os.path.join(gt_folder,   fname))
    lm   = np.load(os.path.join(lm_folder,   fname)).astype(bool)
    tm   = np.load(os.path.join(tm_folder,   fname)).astype(bool)
    gt = crop_back(apply_transform(gt))
    lm = crop_back(apply_transform_to_mask(lm))
    tm = crop_back(apply_transform_to_mask(tm))
    if is_cbct: test = apply_transform(test)
    test = crop_back(test)
    g_mae, g_rmse, g_psnr = compute_mae(test, gt), compute_rmse(test, gt), compute_psnr(test, gt, DATA_RANGE)
    g_ssim, ssim_map = ssim(gt, test, data_range=DATA_RANGE, full=True)
    def masked(fn, A, B, M): return fn(A[M], B[M]) if M.any() else np.nan
    l_mae  = masked(compute_mae,  test, gt, lm)
    l_rmse = masked(compute_rmse, test, gt, lm)
    l_psnr = masked(lambda x,y: compute_psnr(x, y, DATA_RANGE), test, gt, lm)
    l_ssim = np.nanmean(ssim_map[lm]) if lm.any() else np.nan
    t_mae  = masked(compute_mae,  test, gt, tm)
    t_rmse = masked(compute_rmse, test, gt, tm)
    t_psnr = masked(lambda x,y: compute_psnr(x, y, DATA_RANGE), test, gt, tm)
    t_ssim = np.nanmean(ssim_map[tm]) if tm.any() else np.nan
    return (g_mae, g_rmse, g_psnr, g_ssim,
            l_mae, l_rmse, l_psnr, l_ssim,
            t_mae, t_rmse, t_psnr, t_ssim)

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
    for path in files:
        vals = compare_single(os.path.basename(path),
                              test_base if is_cbct else os.path.join(test_base, f"volume-{vol_idx}"),
                              gt_folder, lm_folder, tm_folder, is_cbct)
        for k,v in zip(keys, vals): acc[k].append(v)
    return tuple(np.nanmean(acc[k]) for k in keys)

def run_eval(volumes, base_folder, is_cbct, gt_folder, lm_folder, tm_folder):
    results = {}
    for v in volumes:
        vals = compare_batch(v, base_folder, gt_folder, lm_folder, tm_folder, is_cbct)
        if vals is not None:
            results[v] = vals
    return results

# ──────── global & region eval ─────────────────────────────────────────────────
def print_global_metrics(volumes, eval_sets, results):
    metrics = ["MAE", "RMSE", "PSNR", "SSIM"]
    hdr = "Vol".rjust(4) + " | " + " | ".join(
        " ".join(f"{(label+'_'+m):>10}" for m in metrics)
        for label, _, _ in eval_sets
    )
    print(" GLOBAL METRICS") 
    print(hdr)
    print("-" * len(hdr))
    for v in volumes:
        parts = []
        for label, _, _ in eval_sets:
            vals = results[label].get(v, [np.nan] * len(metrics))
            parts.append(
                " ".join(
                    f"{vals[k]:>10.2f}" if m != "SSIM" else f"{vals[k]:>10.3f}"
                    for k, m in enumerate(metrics)
                )
            )
        print(f"{v:4d} | " + " | ".join(parts))
    overall = lambda rr: np.nanmean(np.stack(list(rr.values()), axis=0), axis=0)
    overall_results = {label: overall(results[label]) for label, _, _ in eval_sets}
    print("-" * len(hdr))
    overall_parts = []
    for label, _, _ in eval_sets:
        vals = overall_results[label]
        overall_parts.append(
            " ".join(
                f"{vals[k]:>10.2f}" if m != "SSIM" else f"{vals[k]:>10.3f}"
                for k, m in enumerate(metrics)
            )
        )
    print(f"{'ALL':>4} | " + " | ".join(overall_parts))


def print_region_metrics(volumes, eval_sets, results):
    metrics = ["MAE", "RMSE", "PSNR", "SSIM"]
    hdr = (
        "Vol".rjust(4) + " | " + "Region".rjust(6) + " | " +
        " | ".join(
            " ".join(f"{(label+'_'+m):>10}" for m in metrics)
            for label, _, _ in eval_sets
        )
    )
    print(" LIVER & TUMOR METRICS") 
    print(hdr)
    print("-" * len(hdr))

    # per‐volume
    for v in volumes:
        for region, base in [("Liver", 4), ("Tumor", 8)]:
            parts = []
            for label, _, _ in eval_sets:
                vals = results[label].get(v, [np.nan]*12)
                parts.append(
                    " ".join(
                        f"{vals[base+k]:>10.2f}" if m != "SSIM" else f"{vals[base+k]:>10.3f}"
                        for k, m in enumerate(metrics)
                    )
                )
            print(f"{v:4d} | {region:>6} | " + " | ".join(parts))

    # separator
    print("-" * len(hdr))

    # overall (mean across volumes)
    for region, base in [("Liver", 4), ("Tumor", 8)]:
        parts = []
        for label, _, _ in eval_sets:
            # collect [vol x metrics] array for this label+region
            arr = np.stack(
                [results[label][v][base:base+len(metrics)] for v in volumes],
                axis=0
            )
            mean_vals = np.nanmean(arr, axis=0)
            parts.append(
                " ".join(
                    f"{mean_vals[k]:>10.2f}" if m != "SSIM" else f"{mean_vals[k]:>10.3f}"
                    for k, m in enumerate(metrics)
                )
            )
        print(f"{'ALL':>4} | {region:>6} | " + " | ".join(parts))

def evaluate_global_and_region(volumes, eval_sets, gt_folder, lm_folder, tm_folder):
    """
    Run global and region-based evaluation and print tables.
    """
    results = {
        label: run_eval(volumes, folder, is_cbct, gt_folder, lm_folder, tm_folder)
        for label, folder, is_cbct in eval_sets
    }
    print_global_metrics(volumes, eval_sets, results)
    print_region_metrics(volumes, eval_sets, results)

# ──────── masked intensity stats ──────────────────────────────────────────────
def compute_masked_stats(img_folder, mask_folders, volumes, eval_sets):
    stats = {label: {} for label, _, _ in [('GT', None, False)] + eval_sets}
    liver_folder, tumor_folder = mask_folders
    for v in volumes:
        stats['GT'][v] = {}
        for mask_name, m_folder in [('Liver', liver_folder), ('Tumor', tumor_folder)]:
            vals = []
            files = get_slice_files(img_folder, v, True)
            for p in files:
                img = apply_transform(np.load(p))
                img = crop_back(img)
                m_raw = np.load(os.path.join(m_folder, os.path.basename(p))).astype(bool)
                m = crop_back(apply_transform_to_mask(m_raw))
                if m.any(): vals.extend(img[m])
            stats['GT'][v][mask_name] = (
                float(np.mean(vals)) if vals else np.nan,
                float(np.std(vals))  if vals else np.nan
            )
        for label, folder, is_cbct in eval_sets:
            stats[label].setdefault(v, {})
            for mask_name, m_folder in [('Liver', liver_folder), ('Tumor', tumor_folder)]:
                vals = []
                files = get_slice_files(folder, v, is_cbct)
                for p in files:
                    img = np.load(p)
                    if is_cbct: img = apply_transform(img)
                    img = crop_back(img)
                    m_raw = np.load(os.path.join(m_folder, os.path.basename(p))).astype(bool)
                    m = crop_back(apply_transform_to_mask(m_raw))
                    if m.any(): vals.extend(img[m])
                stats[label][v][mask_name] = (
                    float(np.mean(vals)) if vals else np.nan,
                    float(np.std(vals))  if vals else np.nan
                )
    return stats

def print_masked_table(volumes, eval_sets, stats):
    header = "Vol | Mask  | " + " | ".join(f"{label}_mean {label}_std" for label,_,_ in [('GT',None,False)]+eval_sets)
    print("\nMASKED INTENSITY STATS (per volume)")
    print(header)
    print("-"*len(header))
    for v in volumes:
        for mask in ['Liver','Tumor']:
            row = f"{v:>3d} | {mask:>5} | " + " | ".join(
                f"{stats[label][v][mask][0]:>6.2f} {stats[label][v][mask][1]:>6.2f}" if not np.isnan(stats[label][v][mask][0]) else "   nan    nan"
                for label,_,_ in [('GT',None,False)]+eval_sets
            )
            print(row)
    print("-"*len(header))
    for mask in ['Liver','Tumor']:
        parts = []
        for label,_,_ in [('GT',None,False)]+eval_sets:
            means = [stats[label][v][mask][0] for v in volumes]
            stds  = [stats[label][v][mask][1] for v in volumes]
            mu = np.nanmean(means)
            sigma = np.nanmean(stds)
            parts.append(f"{mu:>6.2f} {sigma:>6.2f}")
        print(f"ALL | {mask:>5} | " + " | ".join(parts))

# ──────── per-slice stats & printing ──────────────────────────────────────────
SliceStat = namedtuple("SliceStat", [
    "volume","slice_idx",
    "cbct_mae","pred_mae","delta_mae",
    "cbct_ssim","pred_ssim","delta_ssim",
    "tumor_pixels","t_pred_mae"
])

def evaluate_slice_stats(volumes, cbct_base, pred_base, gt_folder, lm_folder, tm_folder):
    stats = collect_slice_stats(volumes, cbct_base, pred_base, gt_folder, lm_folder, tm_folder)
    top5 = {}
    per_vol = lambda attr: {s.volume: max(stats, key=lambda x:getattr(x, attr)) for s in stats}
    top5['delta_mae']  = sorted(per_vol('delta_mae').values(), key=lambda s:s.delta_mae, reverse=True)[:5]
    top5['delta_ssim'] = sorted(per_vol('delta_ssim').values(),key=lambda s:s.delta_ssim, reverse=True)[:5]
    top5['pred_mae']  = sorted(stats, key=lambda s:s.pred_mae, reverse=True)[:5]
    top5['pred_ssim'] = sorted(stats, key=lambda s:s.pred_ssim)[:5]
    filt = [s for s in stats if s.tumor_pixels>300]
    top5['tumor_best'] = sorted(filt, key=lambda s:s.t_pred_mae)[:5]
    print_slice_table("Top 5 ΔMAE (one slice per volume):", top5['delta_mae'], ['cbct_mae','pred_mae','delta_mae'])
    print_slice_table("Top 5 ΔSSIM (one slice per volume):", top5['delta_ssim'], ['cbct_ssim','pred_ssim','delta_ssim'])

    print("\nTop 5 worst Pred MAE (all volumes):")
    print_slice_table("", top5['pred_mae'], ['pred_mae','cbct_mae'])
    print("\nTop 5 worst Pred SSIM (all volumes):")
    print_slice_table("", top5['pred_ssim'], ['pred_ssim','cbct_ssim'])

    print("\nTop 5 best Pred tumor MAE (≥300 tumor px):")
    hdr = "Vol Slice | Tumor_px | Pred_tumor_MAE"
    print(hdr)
    print("-"*len(hdr))
    for s in top5['tumor_best']:
        print(f"{s.volume:>3d} {s.slice_idx:>5d} | {s.tumor_pixels:>8d} | {s.t_pred_mae:>16.3f}")

# ──────── main entrypoint ─────────────────────────────────────────────────────
def main():
    # paths & eval sets
    cbct_base           = os.path.expanduser("~/thesis/training_data/CBCT/test")
    cbct490_base        = os.path.expanduser("~/thesis/training_data/CBCT/scaled-490")

    pred_base           = os.path.expanduser("~/thesis/predictions/v1")
    predspeed_base      = os.path.expanduser("~/thesis/predictions/v1_speed")
    pred490_base        = os.path.expanduser("~/thesis/predictions/v1_490")
    pred490speed_base   = os.path.expanduser("~/thesis/predictions/v1_490_speed")

    v2_pred490speed_base             = os.path.expanduser("~/thesis/predictions/v2_490_speed")
    v2_pred490speed100steps_base     = os.path.expanduser("~/thesis/predictions/v2_490_speed_100steps")
    v2_pred490speedstepsize20_base   = os.path.expanduser("~/thesis/predictions/v2_490_speed_stepsize20")
    v2_pred490speedstepsize20v2_base = os.path.expanduser("~/thesis/predictions/v2_490_speed_stepsize20_v2")
    v2_cbct                          = os.path.expanduser("~/thesis/predictions/predictionsV2-490-50steps_v2_cbct")

    perc_base = os.path.expanduser("~/thesis/predictions/predctions_perceptual_with_float32")

    v3_pred490stepsize20 = os.path.expanduser("~/thesis/predictions/predctions_controlnet_v3")
    v3_pred490stepsize1  = os.path.expanduser("~/thesis/predictions/predictions-v3-stepsize1")

    v4_pred490stepsize20 = os.path.expanduser("~/thesis/predictions/prediction_controlnet_v4")
    v5                   = os.path.expanduser("~/thesis/predictions/predictions_v5")


    v7                   = os.path.expanduser("~/thesis/predictions/predictions_controlnet_v7-data-augmentation")
    v7_260_lin                   = os.path.expanduser("~/thesis/predictions/thesis-ready/256/best-model/50-steps-linear")
    v7_260_pow                   = os.path.expanduser("~/thesis/predictions/thesis-ready/256/best-model/50-steps-power")
    v7_1000                      = os.path.expanduser("~/thesis/predictions/thesis-ready/490/best-model/1000-steps-linear")

    trained_after_joint = os.path.expanduser(
        "~/thesis/predictions/predctions_controlnet_from_unet_trained_after_joint_v2"
    )

    nl                   = os.path.expanduser("~/thesis/predictions/predictions_tanh_v2")
    nl3                  = os.path.expanduser("~/thesis/predictions/predictions_tanh_v3")
    nl5                  = os.path.expanduser("~/thesis/predictions/predictions_tanh_v5")
    nl6 =                  os.path.expanduser("~/thesis/predictions/thesis-ready/490/best-model/50-steps-linear-tanh")

    gt_folder           = os.path.expanduser("~/thesis/training_data/CT/test")
    liver_mask_folder   = os.path.expanduser("~/thesis/training_data/liver/test")
    tumor_mask_folder   = os.path.expanduser("~/thesis/training_data/tumor/test")


    eval_sets = [
        # ("CBCT",  cbct490_base, True),
        ("v7_50",    v7, False),
        ("v7_1000",    v7_1000, False),
    ]

    volumes = VALID_VOLUMES

    # run evaluations
    evaluate_global_and_region(volumes, eval_sets, gt_folder, liver_mask_folder, tumor_mask_folder)

    stats = compute_masked_stats(gt_folder, (liver_mask_folder, tumor_mask_folder), volumes, eval_sets)
    print_masked_table(volumes, eval_sets, stats)

    # evaluate_slice_stats(volumes, cbct_base, pred_base, gt_folder, liver_mask_folder, tumor_mask_folder)

if __name__ == "__main__":
    main()
