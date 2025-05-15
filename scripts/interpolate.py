#!/usr/bin/env python
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

from scipy.ndimage import gaussian_filter1d, convolve1d, median_filter
from scipy.signal import savgol_filter
from evaluation import compute_mae, compute_rmse, compute_psnr, DATA_RANGE, ssim
from torchvision import transforms
from PIL import Image

# Optional advanced methods
try:
    from skimage.restoration import denoise_tv_chambolle, estimate_sigma
except ImportError:
    denoise_tv_chambolle = None
    estimate_sigma     = None

# — GT per-slice pad+resize transform —
transform = transforms.Compose([
    transforms.Pad((0, 64, 0, 64), fill=-1000),
    transforms.Resize((256, 256)),
])

def apply_transform(np_img):
    pil = Image.fromarray(np_img)
    return np.array(transform(pil))

# — Loader —
def load_volume_slices(folder, volume_idx):
    pattern = os.path.join(folder, f"volume-{volume_idx}_slice_*.npy")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No slices found for volume {volume_idx} in {folder}")
    return np.stack([np.load(f) for f in files], axis=0)

# — Evaluation —
def evaluate_volume(interp, gt):
    assert interp.shape == gt.shape, f"Shape mismatch: {interp.shape} vs {gt.shape}"
    maes, rmses, psnrs, ssims = [], [], [], []
    for t, g in zip(interp, gt):
        maes.append(compute_mae(t, g))
        rmses.append(compute_rmse(t, g))
        psnrs.append(compute_psnr(t, g, data_range=DATA_RANGE))
        ssims.append(ssim(g, t, data_range=DATA_RANGE))
    return {
        'MAE': np.mean(maes),
        'RMSE': np.mean(rmses),
        'PSNR': np.mean(psnrs),
        'SSIM': np.mean(ssims)
    }

# — Smoothing methods —
def smooth_z_gaussian(stack, sigma):
    return gaussian_filter1d(stack.astype(np.float32), sigma=sigma, axis=0, mode='nearest')

def smooth_z_box(stack, weights):
    return convolve1d(stack.astype(np.float32), weights, axis=0, mode='nearest')

def smooth_z_median(stack, size):
    return median_filter(stack.astype(np.float32), size=(size,1,1), mode='nearest')

def smooth_z_savgol(stack, window_length, polyorder):
    return savgol_filter(stack.astype(np.float32), window_length=window_length, polyorder=polyorder, axis=0, mode='mirror')

def smooth_z_exponential(stack, alpha):
    out = stack.astype(np.float32).copy()
    for i in range(1, out.shape[0]):
        out[i] = alpha * out[i] + (1 - alpha) * out[i-1]
    return out

def smooth_z_butterworth(stack, cutoff, order):
    Z = stack.shape[0]
    f = np.fft.rfftfreq(Z)
    Hf = 1 / (1 + (f/cutoff)**(2*order))
    freq = np.fft.rfft(stack.astype(np.float32), axis=0)
    return np.fft.irfft(freq * Hf[:, None, None], n=Z, axis=0)

# Joint guided bilateral along Z using CBCT as guidance
def smooth_z_guided_bilateral(stack, guide, sigma_spatial, sigma_range):
    Z, H, W = stack.shape
    out = np.zeros_like(stack, dtype=np.float32)
    radius = int(3 * sigma_spatial)
    for z in range(Z):
        zmin = max(0, z - radius)
        zmax = min(Z, z + radius + 1)
        zs = np.arange(zmin, zmax)
        spatial = np.exp(-((zs - z)**2) / (2 * sigma_spatial**2))
        guide_center = guide[z]
        weights = []
        for w_sp, zi in zip(spatial, zs):
            diff = guide[zi] - guide_center
            range_w = np.exp(-(diff**2) / (2 * sigma_range**2))
            weights.append(w_sp * range_w)
        weights = np.stack(weights, axis=0)
        vals = stack[zmin:zmax] * weights
        out[z] = np.sum(vals, axis=0) / np.sum(weights, axis=0)
    return out

# Optional TV and NL-means
if denoise_tv_chambolle:
    def smooth_tv3d(stack, weight):
        return denoise_tv_chambolle(stack.astype(np.float32), weight=weight)
else:
    smooth_tv3d = None

# — Plotting —
def plot_slice_comparisons(stacks, names, slice_idx):
    ncols = min(4, len(stacks))
    nrows = math.ceil(len(stacks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    gt_slice = stacks[0][slice_idx]
    for i, (stk, name) in enumerate(zip(stacks, names)):
        sl = stk[slice_idx]
        ax = axes[i]
        ax.imshow(sl, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Ground Truth')
        else:
            m = evaluate_volume(sl[np.newaxis], gt_slice[np.newaxis])
            ax.set_title(f"{name}\nMAE {m['MAE']:.2f}, RMSE {m['RMSE']:.2f}\nPSNR {m['PSNR']:.2f}, SSIM {m['SSIM']:.3f}")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

# — Main —
if __name__ == '__main__':
    volume_idx = 33
    slice_idx = 150
    base = os.path.expanduser('~/thesis')
    sct_dir = os.path.join(base, 'predictions', 'predctions_controlnet_v3', f'volume-{volume_idx}')
    gt_dir = os.path.join(base, 'training_data', 'CT', 'test')
    cbct_dir = os.path.join(base, 'training_data', 'CBCT', '490', 'test')

    print('Loading volumes...')
    raw_sct = load_volume_slices(sct_dir, volume_idx)
    gt_full = load_volume_slices(gt_dir, volume_idx)
    cbct_full = load_volume_slices(cbct_dir, volume_idx)
    gt_full = np.stack([apply_transform(im) for im in gt_full], axis=0)
    cbct_full = np.stack([apply_transform(im) for im in cbct_full], axis=0)
    gt = gt_full[: raw_sct.shape[0]]
    cbct = cbct_full[: raw_sct.shape[0]]
    print(f"Shapes: raw {raw_sct.shape}, gt {gt.shape}, cbct {cbct.shape}")

    # parameter grids
    gaussian_sigmas = [1.0, 1.5, 2.0]
    box_weights = {'uniform3': (1/3, 1/3, 1/3), 'tri3': (0.25, 0.5, 0.25), 'wide5': (0.1, 0.2, 0.4, 0.2, 0.1)}
    median_sizes = [3, 5]
    savgol_params = [(5, 2), (7, 3)]
    alphas = [0.1, 0.3, 0.5]
    butter_params = [(0.1, 2), (0.2, 2)]
    tv_weights = [0.05, 0.1] if smooth_tv3d else []
    guided_params = [(1.0, 50.0), (2.0, 25.0)]

    # build and run methods sequentially
    methods = []
    for s in gaussian_sigmas:
        methods.append((partial(smooth_z_gaussian, sigma=s), f'gauss σ={s}'))
    for n, w in box_weights.items():
        methods.append((partial(smooth_z_box, weights=w), f'box {n}'))
    for m in median_sizes:
        methods.append((partial(smooth_z_median, size=m), f'median size={m}'))
    for wl, po in savgol_params:
        methods.append((partial(smooth_z_savgol, window_length=wl, polyorder=po), f'savgol wl={wl},po={po}'))
    for a in alphas:
        methods.append((partial(smooth_z_exponential, alpha=a), f'exp α={a}'))
    for co, or_ in butter_params:
        methods.append((partial(smooth_z_butterworth, cutoff=co, order=or_), f'butter fc={co},o={or_}'))
    for w in tv_weights:
        methods.append((partial(smooth_tv3d, weight=w), f'tv weight={w}'))
    for sp, rg in guided_params:
        methods.append((partial(smooth_z_guided_bilateral, guide=cbct, sigma_spatial=sp, sigma_range=rg), f'guided sp={sp},rg={rg}'))

    results = []
    stacks = [gt, raw_sct, cbct]
    names = ['GT', 'raw', 'cbct']
    results.append({'name': 'raw', 'method': 'raw', 'param': '-', **evaluate_volume(raw_sct, gt)})
    results.append({'name': 'cbct', 'method': 'cbct', 'param': '-', **evaluate_volume(cbct, gt)})

    for idx, (func, name) in enumerate(methods, 1):
        print(f"[{idx}/{len(methods)}] Applying {name}...")
        sm = func(raw_sct)
        res = evaluate_volume(sm, gt)
        stacks.append(sm)
        names.append(name)
        results.append({'name': name, 'method': name.split()[0], 'param': name.split()[-1], **res})

    # compile results into DataFrame
    df = pd.DataFrame(results)
    df_all = df[['name', 'method', 'param', 'MAE', 'RMSE', 'PSNR', 'SSIM']]
    print('Metrics table:')
    print(df_all.to_markdown(index=False))

    # select top 5 (excluding GT/raw/cbct)
    df_best = df_all[df_all['name'] != 'raw'].nsmallest(5, 'MAE')

    selected = ['GT'] + ['raw'] + ['cbct'] + df_best['name'].tolist()

    sel_stacks = [stacks[names.index(n)] for n in selected]

    sel_names  = selected

    print(f"Plotting cbct, GT, raw and top 5 methods...")
    plot_slice_comparisons(sel_stacks, sel_names, slice_idx)
