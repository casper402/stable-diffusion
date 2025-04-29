#!/usr/bin/env python
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import gaussian_filter1d, convolve1d
from evaluation import compute_mae, compute_rmse, compute_psnr, DATA_RANGE, ssim
from torchvision import transforms
from PIL import Image

# — your GT per-slice padding+resize transform —
transform = transforms.Compose([
    transforms.Pad((0, 64, 0, 64), fill=-1000),
    transforms.Resize((256, 256)),
])

def apply_transform(np_img):
    pil = Image.fromarray(np_img)
    return np.array(transform(pil))

def load_volume_slices(folder, volume_idx):
    pattern = os.path.join(folder, f"volume-{volume_idx}_slice_*.npy")
    files   = sorted(glob.glob(pattern))
    stack   = np.stack([np.load(f) for f in files], axis=0)
    return stack

def evaluate_volume(interp, gt):
    assert interp.shape == gt.shape
    maes, rmses, psnrs, ssims = [], [], [], []
    for i in range(interp.shape[0]):
        t, g = interp[i], gt[i]
        maes.append( compute_mae(t, g) )
        rmses.append( compute_rmse(t, g) )
        psnrs.append( compute_psnr(t, g, data_range=DATA_RANGE) )
        ssims.append( ssim(g, t, data_range=DATA_RANGE) )
    return {
        'MAE':  np.mean(maes),
        'RMSE': np.mean(rmses),
        'PSNR': np.mean(psnrs),
        'SSIM': np.mean(ssims),
    }

def smooth_z_gaussian(stack, sigma):
    return gaussian_filter1d(stack.astype(np.float32), sigma=sigma, axis=0, mode='nearest')

def smooth_z_box(stack, weights):
    return convolve1d(stack.astype(np.float32), weights, axis=0, mode='nearest')

def plot_slice_comparisons(stacks, names, slice_idx):
    """
    stacks: list of 3D arrays [gt, raw, sm1, sm2, ...]
    names:  list of strings of same length
    slice_idx: int
    """
    n = len(stacks)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()
    gt_slice = stacks[0][slice_idx]

    for i, (stk, name) in enumerate(zip(stacks, names)):
        sl = stk[slice_idx]
        ax = axes[i]
        ax.imshow(sl, cmap='gray')
        ax.axis('off')

        if i == 0:
            title = "Ground Truth"
        else:
            m = evaluate_volume(np.expand_dims(sl, 0), np.expand_dims(gt_slice, 0))
            title = (
                f"{name}\n"
                f"MAE {m['MAE']:.2f}, RMSE {m['RMSE']:.2f}\n"
                f"PSNR {m['PSNR']:.2f}, SSIM {m['SSIM']:.3f}"
            )
        ax.set_title(title, fontsize=10)

    # turn off any unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Slice {slice_idx}", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    volume_idx = 3
    slice_idx  = 50

    base     = os.path.expanduser("~/thesis")
    sct_dir  = os.path.join(base, "predictions", f"volume-{volume_idx}")
    gt_dir   = os.path.join(base, "training_data", "CT", "test")

    # load stacks
    raw_sct = load_volume_slices(sct_dir, volume_idx)
    gt_full = load_volume_slices(gt_dir, volume_idx)
    # transform & truncate GT
    gt_full = np.stack([apply_transform(im) for im in gt_full], axis=0)
    gt      = gt_full[: raw_sct.shape[0] ]

    # set up smoothers
    gaussian_sigmas = [0.5, 1.0, 2.0, 3.0]
    box_weights = {
        'uniform3': (1/3, 1/3, 1/3),
        'tri3':     (0.25, 0.5, 0.25),
        'wide5':    (0.1, 0.2, 0.4, 0.2, 0.1),
    }

    # collect results
    results = []
    stacks  = [gt, raw_sct]
    names   = ['GT', 'raw']
    results.append({'method':'raw','param':'—', **evaluate_volume(raw_sct, gt)})

    for σ in gaussian_sigmas:
        sm = smooth_z_gaussian(raw_sct, sigma=σ)
        stacks.append(sm)
        names.append(f'gauss σ={σ}')
        results.append({'method':'gaussian','param':σ, **evaluate_volume(sm, gt)})

    for name, w in box_weights.items():
        sm = smooth_z_box(raw_sct, weights=w)
        stacks.append(sm)
        names.append(f'box {name}')
        results.append({'method':'box','param':name, **evaluate_volume(sm, gt)})

    # show metrics table
    df = pd.DataFrame(results)[['method','param','MAE','RMSE','PSNR','SSIM']]
    print(df.to_markdown(index=False))

    # and plot the chosen slice
    plot_slice_comparisons(stacks, names, slice_idx)

if __name__ == "__main__":
    main()
