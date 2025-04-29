#!/usr/bin/env python
import os
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import concurrent.futures

# Fixed full data range for CT Hounsfield units
DATA_RANGE = 2000.0  # from -1000 to 1000

DEBUG = False

def compute_mae(test, gt):
    """MAE over all pixels."""
    assert test.shape == gt.shape, f"Shapes must match, test.shape={test.shape}, gt.shape={gt.shape}"
    return np.mean(np.abs(test - gt))


def compute_rmse(test, gt):
    """RMSE over all pixels."""
    assert test.shape == gt.shape, f"Shapes must match, test.shape={test.shape}, gt.shape={gt.shape}"
    return np.sqrt(np.mean((test - gt)**2))


def compute_psnr(test, gt, data_range):
    """PSNR over all pixels with fixed data range."""
    assert test.shape == gt.shape, f"Shapes must match, test.shape={test.shape}, gt.shape={gt.shape}"
    return psnr(gt, test, data_range=data_range)


def compare_single(fname, test_folder, gt_folder, liver_mask_folder, tumor_mask_folder):
    """
    Load test, GT, liver mask, tumor mask, compute:
      - global MAE, RMSE, PSNR, SSIM
      - liver MAE, RMSE, PSNR, SSIM
      - tumor MAE, RMSE, PSNR, SSIM
    Returns tuple of twelve floats.
    """
    # load data
    test = np.load(os.path.join(test_folder,    fname))
    gt   = np.load(os.path.join(gt_folder,      fname))
    lm   = np.load(os.path.join(liver_mask_folder, fname)).astype(bool)
    tm   = np.load(os.path.join(tumor_mask_folder, fname)).astype(bool)

    # global metrics
    g_mae  = compute_mae(test, gt)
    g_rmse = compute_rmse(test, gt)
    g_psnr = compute_psnr(test, gt, data_range=DATA_RANGE)
    g_ssim, ssim_map = ssim(gt, test, data_range=DATA_RANGE, full=True)

    # masked MAE/RMSE
    def masked(fn, arr1, arr2, mask):
        if not mask.any():
            return np.nan
        return fn(arr1[mask], arr2[mask])

    l_mae  = masked(compute_mae, test, gt, lm)
    l_rmse = masked(compute_rmse, test, gt, lm)
    t_mae  = masked(compute_mae, test, gt, tm)
    t_rmse = masked(compute_rmse, test, gt, tm)

    # masked PSNR using full image range
    l_psnr = masked(lambda t, g: compute_psnr(t, g, data_range=DATA_RANGE), test, gt, lm)
    t_psnr = masked(lambda t, g: compute_psnr(t, g, data_range=DATA_RANGE), test, gt, tm)

    # masked SSIM: average of the full SSIM map
    l_ssim = np.nan if not lm.any() else np.mean(ssim_map[lm])
    t_ssim = np.nan if not tm.any() else np.mean(ssim_map[tm])

    return (
        g_mae, g_rmse, g_psnr, g_ssim,
        l_mae, l_rmse, l_psnr, l_ssim,
        t_mae, t_rmse, t_psnr, t_ssim
    )


def compare_batch(vol_idx, test_folder, gt_folder, liver_mask_folder, tumor_mask_folder):
    """
    For a given volume, run compare_single on all its slices.
    Returns averages: (g_mae, g_rmse, g_psnr, g_ssim,
                       l_mae, l_rmse, l_psnr, l_ssim,
                       t_mae, t_rmse, t_psnr, t_ssim)
    """
    pattern = os.path.join(test_folder, f"volume-{vol_idx}_slice_*.npy")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for volume {vol_idx}")
        return None

    keys = [
        "g_mae","g_rmse","g_psnr","g_ssim",
        "l_mae","l_rmse","l_psnr","l_ssim",
        "t_mae","t_rmse","t_psnr","t_ssim"
    ]
    lists = {k: [] for k in keys}

    for path in files:
        fname = os.path.basename(path)
        vals  = compare_single(fname, test_folder, gt_folder,
                               liver_mask_folder, tumor_mask_folder)
        for key, val in zip(keys, vals):
            lists[key].append(val)

    # compute means, ignoring NaNs
    agg = tuple(np.nanmean(lists[k]) for k in keys)
    return agg


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
    labels = [
        "Global →", "", "", "",
        "Liver  →", "", "", "",
        "Tumor  →", "", "", ""
    ]
    names = ["MAE","RMSE","PSNR(dB)","SSIM"]

    for v in sorted(all_res):
        vals = all_res[v]
        print(f"Volume {v}:")
        for i, region in enumerate(("Global","Liver","Tumor")):
            offs = i*4
            region_vals = vals[offs:offs+4]
            print(f"  {region:6} → " + ", ".join(f"{n} {region_vals[j]:.3f}" for j, n in enumerate(names)))

    # Overall averages
    if all_res:
        arr = np.array(list(all_res.values()))  # (n_vols, 12)
        overall = np.nanmean(arr, axis=0)
        print("\nOverall average across volumes:")
        for i, region in enumerate(("Global","Liver","Tumor")):
            offs = i*4
            region_vals = overall[offs:offs+4]
            print(f"  {region:6} → " + ", ".join(f"{n} {region_vals[j]:.3f}" for j, n in enumerate(names)))
    else:
        print("No volumes to summarize.")

if __name__ == "__main__":
    # ─── CONFIG ────────────────────────────────────────────────
    test_folder        = os.path.expanduser("~/thesis/training_data/CBCT/scaledV2")
    gt_folder          = os.path.expanduser("~/thesis/training_data/CT")
    liver_mask_folder  = os.path.expanduser("~/thesis/training_data/masks/liver")
    tumor_mask_folder  = os.path.expanduser("~/thesis/training_data/masks/tumor")

    volumes = [
        68, 27, 52, 104, 130, 16, 24, 75, 124, 26, 64, 90, 50, 86, 122,
        106, 65, 62, 128, 69, 15, 117, 96, 3, 76, 109, 18, 120, 73, 79,
        83, 14, 58, 17, 112, 13, 110, 125, 1, 126, 93, 51, 107, 91, 85,
        82, 67, 102, 94, 56, 84, 53, 100, 11, 48, 101, 57, 55, 80, 39,
        5, 49, 78, 129, 123, 7, 10, 88, 121, 95, 127, 92, 105, 116, 6,
        19, 115, 97, 2, 118, 66, 54, 25, 63, 108, 22, 113, 8, 111, 114,
        9, 74, 21, 77, 20, 103, 70, 87, 119, 4
    ]

    compare_all_volumes(volumes,
                        test_folder, gt_folder,
                        liver_mask_folder, tumor_mask_folder)
