#!/usr/bin/env python
import os
import glob
import math
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from torchvision import transforms
from PIL import Image

from evaluation import compute_mae, compute_rmse, compute_psnr, DATA_RANGE, ssim


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
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No slices for volume {volume_idx} in {folder}")
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
    return {'MAE': np.mean(maes), 'RMSE': np.mean(rmses), 'PSNR': np.mean(psnrs), 'SSIM': np.mean(ssims)}


# — Advanced dependencies —
try:
    from skimage.restoration import (
        denoise_tv_chambolle,
        estimate_sigma,
        denoise_nl_means,
        denoise_bilateral
    )
except ImportError:
    denoise_tv_chambolle = estimate_sigma = denoise_nl_means = denoise_bilateral = None

try:
    from bm4d import bm4d
except ImportError:
    bm4d = None

try:
    from medpy.filter.smoothing import anisotropic_diffusion
except ImportError:
    anisotropic_diffusion = None


# — Feature toggles —
ENABLE_3D_GAUSS = True
ENABLE_3D_BILATERAL = True
ENABLE_NLM = True
ENABLE_BM4D = True
ENABLE_TV = True
ENABLE_ANISO_DIFFUSION = True


# — 3D smoothing functions —
def smooth_3d_gaussian(stack, sigma_z, sigma_xy):
    return gaussian_filter(stack.astype(np.float32), sigma=(sigma_z, sigma_xy, sigma_xy), mode='nearest')

def smooth_3d_bilateral(stack, sigma_spatial, sigma_range):
    if denoise_bilateral is None:
        raise ImportError("denoise_bilateral not available")
    tmp = np.empty_like(stack, dtype=np.float32)
    win = int(6 * sigma_spatial + 1)
    for z in range(stack.shape[0]):
        tmp[z] = denoise_bilateral(stack[z].astype(np.float32), sigma_color=sigma_range, sigma_spatial=sigma_spatial, channel_axis=None, win_size=win)
    out = np.zeros_like(tmp)
    radius = int(3 * sigma_spatial)
    Z = stack.shape[0]
    for z in range(Z):
        zmin, zmax = max(0, z-radius), min(Z, z+radius+1)
        zs = np.arange(zmin, zmax)
        spatial = np.exp(-((zs-z)**2)/(2*sigma_spatial**2))[:,None,None]
        range_w = np.exp(-((tmp[zs]-tmp[z])**2)/(2*sigma_range**2))
        w = spatial * range_w
        out[z] = (tmp[zs] * w).sum(axis=0) / w.sum(axis=0)
    return out

def smooth_nl_means_3d(stack, patch_size, patch_distance, h_factor, n_jobs=None):
    if denoise_nl_means is None or estimate_sigma is None:
        raise ImportError("nl_means not available")
    Z = stack.shape[0]
    cpu = mp.cpu_count() if n_jobs is None else n_jobs
    chunk = int(math.ceil(Z / cpu))
    intervals = [(i, max(0, i*chunk-patch_distance), min(Z, (i+1)*chunk+patch_distance), i*chunk, min(Z, (i+1)*chunk)) for i in range(cpu)]
    def proc(arg):
        i, s, e, ps, pe = arg
        sub = stack[s:e]
        sigma = estimate_sigma(sub, channel_axis=None)
        den = denoise_nl_means(sub.astype(np.float32), h=h_factor*sigma, patch_size=patch_size, patch_distance=patch_distance, channel_axis=None)
        start = ps - s; end = start + (pe-ps)
        return i, den[start:end]
    with ThreadPool(cpu) as pool:
        parts = pool.map(proc, intervals)
    out = np.zeros_like(stack)
    for idx, block in parts:
        _, s, e, ps, pe = intervals[idx]
        out[ps:pe] = block
    return out

def smooth_bm4d_method(stack, sigma):
    if bm4d is None:
        raise ImportError("bm4d not available")
    return bm4d(stack.astype(np.float32), sigma)

def smooth_tv3d_aniso(stack, wz, wxy):
    if denoise_tv_chambolle is None:
        raise ImportError("TV not available")
    return denoise_tv_chambolle(stack.astype(np.float32), weight=(wz+2*wxy)/3)

def smooth_anisotropic_diffusion(stack, it, kappa):
    if anisotropic_diffusion is None:
        raise ImportError("anisotropic_diffusion not available")
    return anisotropic_diffusion(stack.astype(np.float32), niter=it, kappa=kappa, gamma=0.1)


# — Plotting —
def plot_slice_comparisons(stacks, names, idx):
    cols = min(4, len(stacks)); rows = math.ceil(len(stacks)/cols)
    fig, ax = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    ax = ax.flatten(); gt_slice = stacks[0][idx]
    for i, (stk, name) in enumerate(zip(stacks, names)):
        a = ax[i]; img = stk[idx]
        a.imshow(img, cmap='gray'); a.axis('off')
        if i>0:
            m = evaluate_volume(img[None], gt_slice[None])
            a.set_title(f"{name}\nMAE {m['MAE']:.2f}, PSNR {m['PSNR']:.2f}")
    plt.tight_layout(); plt.show()


# — Main —
if __name__=='__main__':
    volume_idx, slice_idx = 26, 150
    base = os.path.expanduser('~/thesis')
    sct_dir = os.path.join(base, 'predictions', 'predctions_controlnet_v3', f'volume-{volume_idx}')
    gt_dir = os.path.join(base, 'training_data', 'CT', 'test')
    cbct_dir = os.path.join(base, 'training_data', 'CBCT', '490', 'test')
    print('Loading volumes...')
    raw_sct = load_volume_slices(sct_dir, volume_idx)
    gt_full = load_volume_slices(gt_dir, volume_idx)
    cbct_full = load_volume_slices(cbct_dir, volume_idx)
    gt_full = np.stack([apply_transform(im) for im in gt_full], 0)
    cbct_full = np.stack([apply_transform(im) for im in cbct_full], 0)
    gt, cbct = gt_full[:raw_sct.shape[0]], cbct_full[:raw_sct.shape[0]]
    print(f"Shapes: raw {raw_sct.shape}, gt {gt.shape}, cbct {cbct.shape}")

    # — Parameter grids —
    gaussian_3d_params = [(1,1),(1,2),(2,1)]
    bilateral_params   = [(1.0,0.1),(2.0,0.2)]
    nlm_params         = [(3,3,0.8),(5,5,1.0)]
    bm4d_sigmas        = [5]
    tv_aniso_weights   = [(0.2,0.1),(0.5,0.2)]
    ad_params          = [(10,20),(20,30)]

    # — Build methods —
    methods, stacks, names = [], [gt, raw_sct, cbct], ['GT','raw','cbct']

    # 3D Gaussian
    if ENABLE_3D_GAUSS:
        for sz, sxy in gaussian_3d_params:
            methods.append((partial(smooth_3d_gaussian, sigma_z=sz, sigma_xy=sxy), f'3D-gauss z={sz},xy={sxy}'))
    else:
        print("Skipping 3D Gaussian")

    # 3D Bilateral
    if ENABLE_3D_BILATERAL and denoise_bilateral:
        for sp, rg in bilateral_params:
            methods.append((partial(smooth_3d_bilateral, sigma_spatial=sp, sigma_range=rg), f'3D-bilateral sp={sp},rg={rg}'))
    elif not ENABLE_3D_BILATERAL:
        print("Skipping 3D Bilateral (disabled)")
    else:
        print("Skipping 3D Bilateral (no library)")

    # NL-Means 3D
    if ENABLE_NLM and denoise_nl_means and estimate_sigma:
        for ps, pd, h in nlm_params:
            methods.append((partial(smooth_nl_means_3d, patch_size=ps, patch_distance=pd, h_factor=h), f'NLM ps={ps},pd={pd},h={h}'))
    elif not ENABLE_NLM:
        print("Skipping NL-Means (disabled)")
    else:
        print("Skipping NL-Means (no library)")

    # BM4D
    if ENABLE_BM4D:
        if bm4d:
            for sigma in bm4d_sigmas:
                methods.append((partial(smooth_bm4d_method, sigma=sigma), f'BM4D σ={sigma}'))
        else:
            print("BM4D enabled but no library")
    else:
        print("Skipping BM4D (disabled)")

        # Anisotropic TV
    if ENABLE_TV and denoise_tv_chambolle:
        for wz, wxy in tv_aniso_weights:
            # Use lambda to bind parameters, avoiding partial keyword issues
            tv_func = lambda stack, wz=wz, wxy=wxy: smooth_tv3d_aniso(stack, wz, wxy)
            methods.append((tv_func, f'TV-aniso wz={wz},wxy={wxy}'))
    elif not ENABLE_TV:
        print("Skipping TV (disabled)")
    else:
        print("Skipping TV (no library)")

    # Anisotropic diffusion
    if ENABLE_ANISO_DIFFUSION and anisotropic_diffusion:
        for it, kappa in ad_params:
            methods.append((partial(smooth_anisotropic_diffusion, it=it, kappa=kappa), f'AD it={it},kappa={kappa}'))
    elif not ENABLE_ANISO_DIFFUSION:
        print("Skipping anisotropic diffusion (disabled)")
    else:
        print("Skipping anisotropic diffusion (no library)")

    # — Run and evaluate —
    results = []
    if not isinstance(pd, type(__import__('pandas'))):
        import pandas as pd

    results.append({'name':'raw','method':'raw','param':'-', **evaluate_volume(raw_sct, gt)})
    results.append({'name':'cbct','method':'cbct','param':'-', **evaluate_volume(cbct, gt)})

    for idx, (func, name) in enumerate(methods, 1):
        print(f"[{idx}/{len(methods)}] Applying {name}...")
        sm = func(raw_sct)
        res = evaluate_volume(sm, gt)
        stacks.append(sm); names.append(name)
        results.append({'name':name,'method':name.split()[0],'param':name.split()[-1],**res})

    df = pd.DataFrame(results)
    print(df[['name','param','MAE','RMSE','PSNR','SSIM']].to_markdown(index=False))

    df_best = df[df['name']!='raw'].nsmallest(5,'MAE')
    selected = ['GT','raw','cbct',*df_best['name'].tolist()]
    sel_stacks = [stacks[names.index(n)] for n in selected]
    plot_slice_comparisons(sel_stacks, selected, slice_idx)
