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
# Evaluate on all defined volumes
VALID_VOLUMES = list(SLICE_RANGES.keys())  # e.g., [3,8,12,26,32,...,129]
# VALID_VOLUMES = [129]

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

# ──────── metric functions ───────────────────────────────────────────────────
def compute_mae(a, b): return np.nanmean(np.abs(a - b))
def compute_rmse(a, b): return np.sqrt(np.nanmean((a - b)**2))
def compute_psnr(a, b, data_range): return psnr(b, a, data_range=data_range)

# ──────── data loading utility ─────────────────────────────────────────────────
def get_slice_files(folder, vol_idx, is_cbct=False):
    base = folder if is_cbct else os.path.join(folder, f"volume-{vol_idx}")
    pattern = os.path.join(base, f"volume-{vol_idx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    rng = SLICE_RANGES.get(vol_idx)
    if rng:
        start, end = rng
        files = [f for f in files if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end]
    return files

# ──────── per-slice comparison ────────────────────────────────────────────────
def compare_single(fname, test_folder, gt_folder, lm_folder, tm_folder, is_cbct):
    test = np.load(os.path.join(test_folder, fname))
    gt   = np.load(os.path.join(gt_folder,   fname))
    lm   = np.load(os.path.join(lm_folder,   fname)).astype(bool)
    tm   = np.load(os.path.join(tm_folder,   fname)).astype(bool)

    gt   = crop_back(apply_transform(gt))
    lm   = crop_back(apply_transform_to_mask(lm))
    tm   = crop_back(apply_transform_to_mask(tm))
    test = crop_back(apply_transform(test)) if is_cbct else crop_back(test)

    g_mae, g_rmse, g_psnr = compute_mae(test, gt), compute_rmse(test, gt), compute_psnr(test, gt, DATA_RANGE)
    g_ssim, ssim_map = ssim(gt, test, data_range=DATA_RANGE, full=True)

    def masked(fn, A, B, M): return fn(A[M], B[M]) if M.any() else np.nan
    l_mae  = masked(compute_mae, test, gt, lm)
    l_rmse = masked(compute_rmse, test, gt, lm)
    l_psnr = masked(lambda x,y: compute_psnr(x, y, DATA_RANGE), test, gt, lm)
    l_ssim = np.nanmean(ssim_map[lm]) if lm.any() else np.nan
    t_mae  = masked(compute_mae, test, gt, tm)
    t_rmse = masked(compute_rmse, test, gt, tm)
    t_psnr = masked(lambda x,y: compute_psnr(x, y, DATA_RANGE), test, gt, tm)
    t_ssim = np.nanmean(ssim_map[tm]) if tm.any() else np.nan

    return (g_mae, g_rmse, g_psnr, g_ssim,
            l_mae, l_rmse, l_psnr, l_ssim,
            t_mae, t_rmse, t_psnr, t_ssim)

# ──────── batch & global eval ─────────────────────────────────────────────────
def compare_batch(vol_idx, base, gt, lm, tm, is_cbct):
    files = get_slice_files(base, vol_idx, is_cbct)
    if not files:
        print(f"No files for volume {vol_idx}")
        return None
    acc = {i:[] for i in range(12)}
    for f in files:
        vals = compare_single(os.path.basename(f),
                              base if is_cbct else os.path.join(base, f"volume-{vol_idx}"),
                              gt, lm, tm, is_cbct)
        for i,v in enumerate(vals): acc[i].append(v)
    return tuple(np.nanmean(acc[i]) for i in range(12))

def run_eval(vols, base, is_cbct, gt, lm, tm):
    return {v: vals for v in vols if (vals := compare_batch(v, base, gt, lm, tm, is_cbct)) is not None}

def overall(rr):
    arr = np.stack(list(rr.values()))
    return np.nanmean(arr, axis=0)

# ──────── HISTOGRAM (Figure 3) ───────────────────────────────────────────────
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

def plot_hu_distributions(gt_folder, cbct_folder, pred_folder, vols, save_path=None):
    bins = np.linspace(-1000, 1000, 200)
    # Treat CT folder structure like CBCT: direct slices
    cth = compute_hu_histogram(gt_folder, vols, is_cbct=True, bins=bins)
    cbh = compute_hu_histogram(cbct_folder, vols, is_cbct=True, bins=bins)
    pth = compute_hu_histogram(pred_folder, vols, is_cbct=False, bins=bins)
    ctr = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(8, 6))
    if cth.sum() > 0:
        plt.plot(ctr, cth/cth.sum(), label='CT')
    else:
        plt.plot(ctr, cth, label='CT')

    if cbh.sum() > 0:
        plt.plot(ctr, cbh/cbh.sum(), label='CBCT')
    else:
        plt.plot(ctr, cbh, label='CBCT')

    if pth.sum() > 0:
        plt.plot(ctr, pth/pth.sum(), label='sCT')
    else:
        plt.plot(ctr, pth, label='sCT')

    plt.xlabel('Hounsfield Units (HU)')
    plt.ylabel('Normalized Frequency')
    plt.title('Figure 3: HU Distribution Comparison')
    # Zoom into central HU range
    # plt.xlim(-800, 800)
    plt.ylim(0, 0.1)
    plt.legend()
    plt.grid(True)
    plt.show()

# ──────── ROI HU STATS ─────────────────────────────────────────────────────────
def compute_roi_stats(gt_folder, cbct_folder, pred_folder, vols):
    """
    Compute mean ± SD of HU values in CT, CBCT, and sCT for lung, soft tissue, and bone.
    """
    regions = {
        'Lung': (-700, -600),        # typical lung parenchyma
        'Soft tissue': (100, 300),   # includes fat to muscle
        'Bone': (300, 1000),           # trabecular/cortical bone
        'Liver': (54, 66),
        'Kidney': (20, 45),
        'Lymph': (10, 20),
        'Muscle': (35, 55),
        'Fat': (-120, -90)
    }
    rows = []
    for r, (lo, hi) in regions.items():
        ct_vals = []
        cb_vals = []
        pr_vals = []
        voxel_count = 0
        for v in vols:
            ct_files = get_slice_files(gt_folder, v, is_cbct=True)
            for ct_fp in ct_files:
                name = os.path.basename(ct_fp)

                cb_fp = os.path.join(cbct_folder, name)
                pr_fp = os.path.join(pred_folder, f"volume-{v}", name)
                if not (os.path.exists(cb_fp) and os.path.exists(pr_fp)):
                    continue
                ct = crop_back(apply_transform(np.load(ct_fp)))
                cb = crop_back(apply_transform(np.load(cb_fp)))
                pr = crop_back(np.load(pr_fp))
                mask = (ct >= lo) & (ct <= hi)
                if mask.any():
                    vals_ct = ct[mask]
                    ct_vals.append(vals_ct)
                    cb_vals.append(cb[mask])
                    pr_vals.append(pr[mask])
                    voxel_count += vals_ct.size
        if voxel_count > 0:
            cts = np.concatenate(ct_vals)
            cbs = np.concatenate(cb_vals)
            prs = np.concatenate(pr_vals)
            rows.append({
                'Region': r,
                'CT mean±sd': f"{cts.mean():.1f} ± {cts.std():.1f}",
                'CBCT mean±sd': f"{cbs.mean():.1f} ± {cbs.std():.1f}",
                'sCT mean±sd': f"{prs.mean():.1f} ± {prs.std():.1f}",
                'Voxel count': voxel_count
            })
        else:
            rows.append({
                'Region': r,
                'CT mean±sd': 'n/a',
                'CBCT mean±sd': 'n/a',
                'sCT mean±sd': 'n/a',
                'Voxel count': 0
            })
    return pd.DataFrame(rows)

# ──────── GLOBAL METRICS TABLE (Table 3) ────────────────────────────────────── (Table 3) ────────────────────────────────────── (Table 3) ────────────────────────────────────── (Table 3) ──────────────────────────────────────
def generate_metrics_table(results, method_name='sCT (GAN)', metrics=['MAE','RMSE','PSNR','SSIM']):
    overall_vals = overall(results)
    row = {'Method': method_name}
    for i, m in enumerate(metrics):
        row[m] = overall_vals[i]
    df = pd.DataFrame([row])
    print("\nTable 3: Global Metrics Comparison")
    print(df.to_markdown(index=False))

# ──────── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vols = VALID_VOLUMES

    # Update these paths as needed
    gt_folder         = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder       = os.path.expanduser("~/thesis/training_data/CBCT/test")
    liver_mask_folder = os.path.expanduser("~/thesis/training_data/liver/test")
    tumor_mask_folder = os.path.expanduser("~/thesis/training_data/tumor/test")
    pred_folder       = os.path.expanduser("/Users/Niklas/thesis/predictions/predctions_controlnet_v3")

    # 2) Table 3: Global metrics
    # results = run_eval(vols, pred_folder, False, gt_folder, liver_mask_folder, tumor_mask_folder)
    # method_label = os.path.basename(pred_folder)
    # generate_metrics_table(results, method_name=method_label)

    # 3) ROI HU stats table
    df_roi = compute_roi_stats(gt_folder, cbct_folder, pred_folder, vols)
    print("\nROI HU Mean±SD (Table X):")
    print(df_roi.to_markdown(index=False))

    # 1) Figure 3: HU histograms
    plot_hu_distributions(gt_folder, cbct_folder, pred_folder, vols)
