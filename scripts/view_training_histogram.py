#!/usr/bin/env python
import os
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from concurrent.futures import ProcessPoolExecutor, as_completed

# ────────────────────────────────────────────────────────────────────────────────
#   CONSTANTS AND TRANSFORMS FOR CT/CBCT
# ────────────────────────────────────────────────────────────────────────────────
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256
_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))

# Transform chain for CBCT (pad + resize)
gt_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W)),
])

def apply_transform(img_np):
    """
    Pad & resize a slice; return as NumPy array.
    """
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    out = gt_transform(t)
    return out.squeeze(0).numpy()

def crop_back(arr):
    """
    Remove padding from 256×256 → 166×256.
    """
    return arr[TOP_CROP:RES_H - BOTTOM_CROP, LEFT_CROP:RES_W - RIGHT_CROP]

# ────────────────────────────────────────────────────────────────────────────────
#   LOAD MANIFEST AND PREPARE FILE LISTS
# ────────────────────────────────────────────────────────────────────────────────
manifest_csv = os.path.expanduser("~/thesis/manifest-filtered.csv")
df = pd.read_csv(manifest_csv)

# Paths from manifest
ct_all           = df['ct_path'].tolist()
cbct_all         = df['cbct_490_path'].tolist()

# Split into train/test
ct_train         = df[df['split']=='train']['ct_path'].tolist()
ct_test          = df[df['split']=='test']['ct_path'].tolist()
cbct_train       = df[df['split']=='train']['cbct_490_path'].tolist()
cbct_test        = df[df['split']=='test']['cbct_490_path'].tolist()

# Volume-35 subsets
ct_vol35         = df[df['ct_path'].str.contains('volume-35')]['ct_path'].tolist()
cbct_vol35       = df[df['cbct_490_path'].str.contains('volume-35')]['cbct_490_path'].tolist()

# ────────────────────────────────────────────────────────────────────────────────
#   PARAMETERS
# ────────────────────────────────────────────────────────────────────────────────
# Limit the number of slices per group (None = all)
SLICE_LIMIT = None  # e.g. 100
# Number of parallel workers; default to CPU core count for efficiency
NUM_WORKERS = os.cpu_count() or 4
print("NUM_WORKERS:", NUM_WORKERS)

# ────────────────────────────────────────────────────────────────────────────────
#   WORKER FUNCTION FOR SINGLE HISTOGRAM
# ────────────────────────────────────────────────────────────────────────────────
def _hist_for_file(fp: str, bins: np.ndarray, is_cbct: bool) -> np.ndarray:
    """
    Load a single slice, transform if needed, crop, and compute raw histogram counts.
    """
    data = np.load(fp)
    if is_cbct:
        data = apply_transform(data)
    arr = crop_back(data)
    return np.histogram(arr.flatten(), bins=bins)[0]

# ────────────────────────────────────────────────────────────────────────────────
#   HISTOGRAM COMPUTATION WITH PARALLEL WORKERS
# ────────────────────────────────────────────────────────────────────────────────
def compute_histograms(groups, bins=np.linspace(-1000,1000,200), limit=None, workers=1):
    """
    Compute normalized histograms for each group of file lists.
    If limit is set, restrict to first `limit` slices; use `workers` for parallel processing.
    Returns dict of {label: (centers, normalized_counts)}.
    """
    results = {}
    centers = (bins[:-1] + bins[1:]) / 2
    for label, file_list, is_cbct in groups:
        files = file_list[:limit] if limit is not None else file_list
        hist = np.zeros(len(bins)-1, dtype=np.float64)
        if workers and workers > 1:
            # Parallel computation
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_hist_for_file, fp, bins, is_cbct) for fp in files]
                for f in as_completed(futures):
                    hist += f.result()
        else:
            # Sequential fallback
            for fp in files:
                hist += _hist_for_file(fp, bins, is_cbct)
        norm = hist / hist.sum() if hist.sum() > 0 else hist
        results[label] = (centers, norm)
    return results

if __name__ == '__main__':
    # Define groups to compute
    groups = [
        ('CT Train',    ct_train,    True),
        ('CT Test',     ct_test,     True),
        ('CBCT Train',  cbct_train,  True),
        ('CBCT Test',   cbct_test,   True),
        ('CT Vol-35',   ct_vol35,    True),
        ('CBCT Vol-35', cbct_vol35,  True),
    ]

    # Compute histograms with optional slice limit and parallel workers
    hist_data = compute_histograms(groups, limit=SLICE_LIMIT, workers=NUM_WORKERS)

    # Optionally include predictions
    pred_folders = []  # e.g. [('sCT', '/path/to/predictions')]
    if pred_folders:
        pred_groups = [(label,
                        [os.path.join(folder, os.path.basename(p)) for p in ct_all],
                        False)
                       for label, folder in pred_folders]
        pred_data = compute_histograms(pred_groups, limit=SLICE_LIMIT, workers=NUM_WORKERS)
        hist_data.update(pred_data)

    # Save to disk
    out_file = os.path.expanduser('~/thesis/data/hu_histograms.npz')
    np.savez(out_file,
             **{f"{lbl}_x": data[0] for lbl, data in hist_data.items()},
             **{f"{lbl}_y": data[1] for lbl, data in hist_data.items()})
    print(f"Saved histogram data to: {out_file} (limit={SLICE_LIMIT}, workers={NUM_WORKERS})")
