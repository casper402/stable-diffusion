#!/usr/bin/env python
import os
import glob
import numpy as np
import concurrent.futures

DEBUG = False

def compute_mae(test, gt):
    """MAE over all pixels."""
    assert test.shape == gt.shape, "Shapes must match"
    return np.mean(np.abs(test - gt))

def compute_rmse(test, gt):
    """RMSE over all pixels."""
    assert test.shape == gt.shape, "Shapes must match"
    return np.sqrt(np.mean((test - gt)**2))

def compute_mae_masked(test, gt, mask):
    """MAE over pixels where mask==1; returns NaN if mask empty."""
    assert test.shape == gt.shape == mask.shape, "Shapes must match"
    m = mask.astype(bool)
    if not m.any():
        return np.nan
    return np.mean(np.abs(test[m] - gt[m]))

def compute_rmse_masked(test, gt, mask):
    """RMSE over pixels where mask==1; returns NaN if mask empty."""
    assert test.shape == gt.shape == mask.shape, "Shapes must match"
    m = mask.astype(bool)
    if not m.any():
        return np.nan
    return np.sqrt(np.mean((test[m] - gt[m])**2))

def compare_single(fname, test_folder, gt_folder, liver_mask_folder, tumor_mask_folder):
    """
    Load test, GT, liver mask, tumor mask, compute:
      - global MAE, global RMSE
      - liver MAE, liver RMSE
      - tumor MAE, tumor RMSE
    Returns tuple of six floats.
    """
    test = np.load(os.path.join(test_folder,    fname))
    gt   = np.load(os.path.join(gt_folder,      fname))
    lm   = np.load(os.path.join(liver_mask_folder, fname)).astype(bool)
    tm   = np.load(os.path.join(tumor_mask_folder, fname)).astype(bool)

    g_mae  = compute_mae(test, gt)
    g_rmse = compute_rmse(test, gt)

    l_mae  = compute_mae_masked(test, gt, lm)
    l_rmse = compute_rmse_masked(test, gt, lm)

    t_mae  = compute_mae_masked(test, gt, tm)
    t_rmse = compute_rmse_masked(test, gt, tm)

    return g_mae, g_rmse, l_mae, l_rmse, t_mae, t_rmse

def compare_batch(vol_idx, test_folder, gt_folder, liver_mask_folder, tumor_mask_folder):
    """
    For a given volume, run compare_single on all its slices.
    Returns averages: (g_mae, g_rmse, l_mae, l_rmse, t_mae, t_rmse)
    """
    pattern = os.path.join(test_folder, f"volume-{vol_idx}_slice_*.npy")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for volume {vol_idx}")
        return None

    lists = {k: [] for k in ("g_mae","g_rmse","l_mae","l_rmse","t_mae","t_rmse")}

    for path in files:
        fname = os.path.basename(path)
        vals  = compare_single(fname, test_folder, gt_folder,
                               liver_mask_folder, tumor_mask_folder)
        for key, val in zip(lists, vals):
            lists[key].append(val)

    # compute means, ignoring NaNs for masked
    return (
        np.mean(lists["g_mae"]),
        np.mean(lists["g_rmse"]),
        np.nanmean(lists["l_mae"]),
        np.nanmean(lists["l_rmse"]),
        np.nanmean(lists["t_mae"]),
        np.nanmean(lists["t_rmse"]),
    )

def compare_all_volumes(volumes, test_folder, gt_folder, liver_mask_folder, tumor_mask_folder):
    all_res = {}

    with concurrent.futures.ThreadPoolExecutor() as exe:
        futs = {
            exe.submit(compare_batch, v, test_folder, gt_folder, liver_mask_folder, tumor_mask_folder): v
            for v in volumes
        }
        for fut in concurrent.futures.as_completed(futs):
            v = futs[fut]
            try:
                res = fut.result()
                if res is not None:
                    all_res[v] = res
                    print(f"Finished volume {v}")
            except Exception as e:
                print(f"Volume {v} error: {e}")

    # Per-volume report
    for v in sorted(all_res):
        g_mae, g_rmse, l_mae, l_rmse, t_mae, t_rmse = all_res[v]
        print(
            f"Volume {v}:\n"
            f"  Global → MAE {g_mae:.3f}, RMSE {g_rmse:.3f}\n"
            f"  Liver  → MAE {l_mae:.3f}, RMSE {l_rmse:.3f}\n"
            f"  Tumor  → MAE {t_mae:.3f}, RMSE {t_rmse:.3f}"
        )

    # Overall averages
    if all_res:
        keys = list(all_res.values())
        agg = np.array(keys)  # shape: (n_volumes, 6)
        print("\nOverall average across volumes:")
        print(f"  Global MAE  = {np.mean(agg[:,0]):.3f}")
        print(f"  Global RMSE = {np.mean(agg[:,1]):.3f}")
        print(f"  Liver  MAE  = {np.nanmean(agg[:,2]):.3f}")
        print(f"  Liver  RMSE = {np.nanmean(agg[:,3]):.3f}")
        print(f"  Tumor  MAE  = {np.nanmean(agg[:,4]):.3f}")
        print(f"  Tumor  RMSE = {np.nanmean(agg[:,5]):.3f}")
    else:
        print("No volumes to summarize.")

if __name__ == "__main__":
    # ─── CONFIG ────────────────────────────────────────────────
    test_folder        = os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/scaledV2")
    gt_folder          = os.path.expanduser("/Users/Niklas/thesis/training_data/CT")
    liver_mask_folder  = os.path.expanduser("/Users/Niklas/thesis/training_data/masks/liver")
    tumor_mask_folder  = os.path.expanduser("/Users/Niklas/thesis/training_data/masks/tumor")

    volumes = [
        68, 27, 52, 104, 130, 16, 24, 75, 124, 26, 64, 90, 50, 86, 122,
        106, 65, 62, 128, 69, 15, 117, 96, 3, 76, 109, 18, 120, 73, 79,
        83, 14, 58, 17, 112, 13, 110, 125, 1, 126, 93, 51, 107, 91, 85,
        82, 67, 102, 94, 56, 84, 53, 100, 11, 48, 101, 57, 55, 80, 39,
        5, 49, 78, 129, 123, 7, 10, 88, 121, 95, 127, 92, 105, 116, 6,
        19, 115, 97, 2, 118, 66, 54, 25, 63, 108, 22, 113, 8, 111, 114,
        9, 74, 21, 77, 20, 103, 70, 87, 119, 4
    ]

    # ──────────────────────────────────────────────────────────

    compare_all_volumes(volumes,
                        test_folder, gt_folder,
                        liver_mask_folder, tumor_mask_folder)
