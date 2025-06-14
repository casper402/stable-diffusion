import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ──────── constants & transforms ─────────────────────────────────────────────
DATA_RANGE = 2000.0    # CT range -1000…1000
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

# Compute padded dimensions and crop offsets
_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))

# Slice ranges per volume (inclusive); None means use all slices
SLICE_SELECT = {
    3: None,
    8: (0, 354),
    12: (0, 320),
    26: None,
    32: (69, 269),
    33: (59, 249),
    35: (91, 268),
    # 54: [0, 4, 11, 19, 26, 33, 40, 48, 55, 62, 70, 77, 84, 91, 99, 106, 113, 120, 128, 135, 142, 149, 157, 164, 171, 179, 186, 193, 200, 208, 215, 222, 229, 237, 244, 251, 259, 266, 273, 280, 2888, 295, 317, 324],
    54: (0, 330),
    59: (0, 311),
    61: (0, 315),
    106: None,
    116: None,
    129: (5, 346)
}
VOLUMES = list(SLICE_SELECT.keys())
VOLUMES = [8, 3]

# Plot configuration
# Available types: 'profile', 'qq', 'hist', 'ba', 'ssim'
PLOT_TYPES = [
    'profile',
    'qq',
    'hist',
    'ba',
    'ssim',
    'slice_metrics',
]
QQ_QUANTILES = 300     # Number of quantiles for Q-Q plots
HIST_BINS    = 100     # Number of bins for histograms
BA_SUBSAMPLE = 500   # Number of points to sample in Bland-Altman plots

# Executors & cache for preload
bg_executor   = ThreadPoolExecutor(max_workers=2)
_preload_cache = {}  # maps volume -> {'future', 'start_time'}

# Transform pipeline for CT/CBCT
gt_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W), interpolation=InterpolationMode.BILINEAR),
])

# Utility: pad & resize
def apply_transform(img_np: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    return gt_transform(t).squeeze(0).numpy()

# Utility: crop back to original aspect
def crop_back(arr: np.ndarray) -> np.ndarray:
    return arr[TOP_CROP:RES_H-BOTTOM_CROP, LEFT_CROP:RES_W-RIGHT_CROP]

# Load .npy slice and prepare
def load_and_prepare(path: str, is_cbct: bool = True) -> np.ndarray:
    data = np.load(path)
    if is_cbct:
        data = apply_transform(data)
    return crop_back(data)

def list_slices(volume: int, folder: str) -> list:
    """
    List slice filenames for a given volume,
    filtering by either a tuple range or explicit list.
    """
    pattern = os.path.join(folder, f"volume-{volume}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    selector = SLICE_SELECT.get(volume)
    # If None, return all
    if selector is None:
        return [os.path.basename(f) for f in files]
    # If explicit list, filter those indices
    if isinstance(selector, list):
        valid = set(selector)
        filtered = [f for f in files
                    if int(os.path.basename(f).split('_')[-1].split('.')[0]) in valid]
    # If tuple, filter inclusive range
    elif isinstance(selector, tuple) and len(selector) == 2:
        start, end = selector
        filtered = [f for f in files
                    if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end]
    else:
        filtered = files
    return [os.path.basename(f) for f in filtered]

# Parallel single-slice loader
def _load_slice_values(args):
    volume, slice_name, gt_folder, cbct_folder, pred_folders = args
    ct   = load_and_prepare(os.path.join(gt_folder, slice_name), is_cbct=True)
    cbct = load_and_prepare(os.path.join(cbct_folder, slice_name), is_cbct=True)
    preds_flat = {}
    for label, folder in pred_folders.items():
        pred_arr = load_and_prepare(os.path.join(folder, f"volume-{volume}", slice_name), is_cbct=False)
        preds_flat[label] = pred_arr.flatten()
    return ct.flatten(), cbct.flatten(), preds_flat

# Parallel full-volume loader
def _load_full_volume(volume, gt_folder, cbct_folder, pred_folders):
    slices = list_slices(volume, gt_folder)
    ct_list, cbct_list = [], []
    preds_list = {label: [] for label in pred_folders}
    with ProcessPoolExecutor() as pexec:
        args = [(volume, sn, gt_folder, cbct_folder, pred_folders) for sn in slices]
        for ct_flat, cbct_flat, pred_dict in pexec.map(_load_slice_values, args):
            ct_list.append(ct_flat)
            cbct_list.append(cbct_flat)
            for lbl, arr in pred_dict.items():
                preds_list[lbl].append(arr)
    return (np.concatenate(ct_list), np.concatenate(cbct_list),
            {lbl: np.concatenate(arrs) for lbl, arrs in preds_list.items()})

# Schedule preload of volume (non-blocking)
def preload_volume(volume, gt_folder, cbct_folder, pred_folders):
    if volume in _preload_cache:
        return
    print(f"[Preload] Starting preload for volume {volume}...")
    start_time = time.time()
    future = bg_executor.submit(_load_full_volume, volume, gt_folder, cbct_folder, pred_folders)
    _preload_cache[volume] = {'future': future, 'start_time': start_time}

# Plot vertical profile line
def plot_profile(slice_name, x_coord, ct_img, cbct_img, preds, volume):
    H, W = ct_img.shape
    x = x_coord if x_coord is not None else random.randint(0, W-1)
    profs = {'CT': profile_line(ct_img, (0, x), (H-1, x), mode='constant', cval=np.nan),
             'CBCT': profile_line(cbct_img, (0, x), (H-1, x), mode='constant', cval=np.nan)}
    profs.update({lbl: profile_line(img, (0, x), (H-1, x), mode='constant', cval=np.nan)
                  for lbl, img in preds.items()})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(ct_img, cmap='gray', vmin=-1000, vmax=1000)
    ax1.axvline(x=x, color='r', lw=2); ax1.axis('off'); ax1.set_title(f'Vol {volume}, {slice_name}')
    for lbl, prof in profs.items(): ax2.plot(prof, label=lbl)
    ax2.set(xlabel='Pixel position', ylabel='HU', title=f'Profile at x={x}')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()

def plot_profile_horizontal(slice_name, y_coord, ct_img, cbct_img, preds, volume):
    H, W = ct_img.shape
    # choose random row if not specified
    y = y_coord if y_coord is not None else random.randint(0, H-1)
    
    # sample each image along the horizontal line y
    profs = {
        'CT':   profile_line(ct_img,  (y, 0), (y, W-1), mode='constant', cval=np.nan),
        'CBCT': profile_line(cbct_img,(y, 0), (y, W-1), mode='constant', cval=np.nan),
    }
    # add any prediction profiles
    for lbl, img in preds.items():
        profs[lbl] = profile_line(img, (y, 0), (y, W-1), mode='constant', cval=np.nan)
    
    # plot image + line and the profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(ct_img, cmap='gray', vmin=-1000, vmax=1000)
    ax1.axhline(y=y, color='r', lw=2)
    ax1.axis('off')
    ax1.set_title(f'Vol {volume}, {slice_name}')
    
    for lbl, prof in profs.items():
        ax2.plot(prof, label=lbl)
    ax2.set(
        xlabel='Column (pixel index)',
        ylabel='HU',
        title=f'Profile at y={y}'
    )
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot Q-Q
def plot_qq(volume, gt_folder, cbct_folder, pred_folders):
    cache = _preload_cache.get(volume)
    if cache:
        print(f"[QQ] Cache hit for volume {volume}")
        start_wait = time.time()
        ct_all, cbct_all, preds_all = cache['future'].result()
        wait_time = time.time() - start_wait
        total_time = wait_time + (start_wait - cache['start_time'])
        print(f"[QQ] Ready in {total_time:.2f}s (wait {wait_time:.2f}s)")
    else:
        print(f"[QQ] Cache miss for volume {volume}")
        t0 = time.time()
        ct_all, cbct_all, preds_all = _load_full_volume(volume, gt_folder, cbct_folder, pred_folders)
        print(f"[QQ] Load in {time.time() - t0:.2f}s")
    qs = np.linspace(0, 1, QQ_QUANTILES)
    ct_q = np.quantile(ct_all, qs)
    targets = [('CBCT', cbct_all)] + list(preds_all.items())
    fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 6))
    axes = np.atleast_1d(axes)
    for ax, (lbl, arr) in zip(axes, targets):
        arr_q = np.quantile(arr, qs)
        ax.plot(ct_q, arr_q, 'o', markersize=4)
        mn, mx = ct_q.min(), ct_q.max()
        ax.plot([mn, mx], [mn, mx], 'k--')
        ax.set(title=f'Q-Q CT vs {lbl}', xlabel='CT quantiles', ylabel=f'{lbl} quantiles')
        ax.grid(True)
    plt.tight_layout(); plt.show()

# Plot histogram of errors
def plot_hist(volume, gt_folder, cbct_folder, pred_folders):
    cache = _preload_cache.get(volume)
    if cache:
        print(f"[Hist] Cache hit for volume {volume}")
        ct_all, cbct_all, preds_all = cache['future'].result()
    else:
        print(f"[Hist] Cache miss for volume {volume}")
        ct_all, cbct_all, preds_all = _load_full_volume(volume, gt_folder, cbct_folder, pred_folders)
    diffs = {'CBCT': cbct_all - ct_all}
    diffs.update({lbl: arr - ct_all for lbl, arr in preds_all.items()})
    fig, axes = plt.subplots(1, len(diffs), figsize=(6*len(diffs), 6))
    axes = np.atleast_1d(axes)
    for ax, (lbl, d) in zip(axes, diffs.items()):
        ax.hist(d, bins=HIST_BINS, alpha=0.7)
        ax.set(title=f'Hist Δ HU: {lbl}', xlabel='Δ HU', ylabel='Count')
        ax.grid(True)
    plt.tight_layout(); plt.show()

# Plot Bland-Altman (Difference vs Mean and vs CT) over full volume
def plot_bland_altman(volume, gt_folder, cbct_folder, pred_folders):
    cache = _preload_cache.get(volume)
    if cache:
        print(f"[BA] Cache hit for volume {volume}")
        ct_all, cbct_all, preds_all = cache['future'].result()
    else:
        print(f"[BA] Cache miss for volume {volume}")
        ct_all, cbct_all, preds_all = _load_full_volume(volume, gt_folder, cbct_folder, pred_folders)
    targets = [('CBCT', cbct_all)] + list(preds_all.items())
    n = len(targets)
    fig, axes = plt.subplots(2, n, figsize=(6*n, 12))
    for idx, (lbl, arr) in enumerate(targets):
        mean_vals = (ct_all + arr) / 2.0
        diff_vals = arr - ct_all
        cnt = mean_vals.size
        if cnt > BA_SUBSAMPLE:
            sel = np.random.choice(cnt, BA_SUBSAMPLE, replace=False)
            m1, d1 = mean_vals[sel], diff_vals[sel]
            m2, d2 = ct_all[sel], diff_vals[sel]
        else:
            m1, d1 = mean_vals, diff_vals
            m2, d2 = ct_all, diff_vals
        md = np.mean(diff_vals)
        sd = np.std(diff_vals)
        lo = md - 1.96*sd; hi = md + 1.96*sd
        ax1 = axes[0, idx]
        ax1.scatter(m1, d1, s=2, alpha=0.3)
        ax1.axhline(md, color='k', linestyle='-'); ax1.text(1000, md, f"Mean={md:.1f}", va='bottom', ha='right')
        ax1.axhline(hi, color='k', linestyle='--'); ax1.text(1000, hi, f"+1.96SD={hi:.1f}", va='bottom', ha='right')
        ax1.axhline(lo, color='k', linestyle='--'); ax1.text(1000, lo, f"-1.96SD={lo:.1f}", va='top', ha='right')
        ax1.set_xlim(-1000, 1000); ax1.set_ylim(-1000, 1000)
        ax1.set(title=f'BA vs Mean: CT vs {lbl}', xlabel='Mean HU', ylabel='Diff HU')
        ax1.grid(True)
        ax2 = axes[1, idx]
        ax2.scatter(m2, d2, s=2, alpha=0.3)
        ax2.axhline(md, color='k', linestyle='-'); ax2.text(1000, md, f"Mean={md:.1f}", va='bottom', ha='right')
        ax2.axhline(hi, color='k', linestyle='--'); ax2.text(1000, hi, f"+1.96SD={hi:.1f}", va='bottom', ha='right')
        ax2.axhline(lo, color='k', linestyle='--'); ax2.text(1000, lo, f"-1.96SD={lo:.1f}", va='top', ha='right')
        ax2.set_xlim(-1000, 1000); ax2.set_ylim(-1000, 1000)
        ax2.set(title=f'BA vs CT: CT vs {lbl}', xlabel='CT HU', ylabel='Diff HU')
        ax2.grid(True)
    plt.tight_layout(); plt.show()

# Plot Spatial SSIM Maps for a given slice
# Top row: GT, CT, and each comparison image
# Second row: local SSIM heatmaps
# Third row: absolute error (MAE)
# Fourth row: HU difference map
def plot_ssim(slice_name, gt_folder, cbct_folder, pred_folders, volume):
    ct_img   = load_and_prepare(os.path.join(gt_folder, slice_name), is_cbct=True)
    cbct_img = load_and_prepare(os.path.join(cbct_folder, slice_name), is_cbct=True)
    preds    = {lbl: load_and_prepare(os.path.join(folder, f"volume-{volume}", slice_name), is_cbct=False)
                for lbl, folder in pred_folders.items()}
    targets = [('GT', ct_img), ('CBCT', cbct_img)] + list(preds.items())
    n = len(targets)
    fig, axes = plt.subplots(4, n, figsize=(6*n, 16))

    for idx, (lbl, img) in enumerate(targets):
        # Row 1: original image
        ax0 = axes[0, idx]
        ax0.imshow(img, cmap='gray', vmin=-1000, vmax=1000)
        ax0.axis('off'); ax0.set_title(lbl)
        # Row 2: SSIM map
        ax1 = axes[1, idx]
        _, s_map = ssim(ct_img, img, full=True, data_range=DATA_RANGE)
        im1 = ax1.imshow(s_map, vmin=0, vmax=1)
        ax1.axis('off'); ax1.set_title(f'SSIM GT vs {lbl}')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        # Row 3: MAE map
        ax2 = axes[2, idx]
        mae_map = np.abs(img - ct_img)
        im2 = ax2.imshow(mae_map, cmap='hot', vmax=500)
        ax2.axis('off'); ax2.set_title(f'MAE GT vs {lbl}')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
        # Row 4: HU diff map
        ax3 = axes[3, idx]
        diff_map = img - ct_img
        im3 = ax3.imshow(diff_map, cmap='gray', vmin=-500, vmax=500)
        ax3.axis('off'); ax3.set_title(f'Diff GT vs {lbl}')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    plt.tight_layout(); plt.show()


def plot_slice_metrics(volume, gt_folder, cbct_folder, pred_folders):
    slices = list_slices(volume, gt_folder)
    # preallocate
    mae_ct_cbct = []
    mae_preds   = {lbl: [] for lbl in pred_folders}
    ssim_ct_cbct = []
    ssim_preds   = {lbl: [] for lbl in pred_folders}

    for slice_name in slices:
        # load imgs
        ct_img   = load_and_prepare(os.path.join(gt_folder, slice_name), True)
        cbct_img = load_and_prepare(os.path.join(cbct_folder, slice_name), True)
        # MAE
        mae_ct_cbct.append(np.mean(np.abs(cbct_img - ct_img)))
        for lbl, folder in pred_folders.items():
            pred    = load_and_prepare(os.path.join(folder, f"volume-{volume}", slice_name), False)
            mae_preds[lbl].append(np.mean(np.abs(pred - ct_img)))
        # SSIM (global, mean over map)
        ssim_ct_cbct.append(ssim(ct_img, cbct_img, data_range=DATA_RANGE))
        for lbl, folder in pred_folders.items():
            pred    = load_and_prepare(os.path.join(folder, f"volume-{volume}", slice_name), False)
            ssim_preds[lbl].append(ssim(ct_img, pred, data_range=DATA_RANGE))

    x = np.arange(len(slices))
    fig, (ax_mae, ax_ssim) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: MAE
    ax_mae.plot(x, mae_ct_cbct, label='CBCT')
    for lbl, vals in mae_preds.items():
        ax_mae.plot(x, vals, label=lbl)
    ax_mae.set(ylabel='MAE (HU)', title=f'Slice-wise MAE & SSIM – Volume {volume}')
    ax_mae.legend(); ax_mae.grid(True)

    # Bottom: SSIM
    ax_ssim.plot(x, ssim_ct_cbct, label='CBCT')
    for lbl, vals in ssim_preds.items():
        ax_ssim.plot(x, vals, label=lbl)
    ax_ssim.set(xlabel='Slice index', ylabel='SSIM')
    ax_ssim.legend(); ax_ssim.grid(True)

    plt.tight_layout()
    plt.show()


# Dispatcher
def plot_multi(slice_name, x_coord, gt_folder, cbct_folder, pred_folders, volume, plot_types=None):
    if plot_types is None:
        plot_types = PLOT_TYPES
    if 'profile' in plot_types:
        ct = load_and_prepare(os.path.join(gt_folder, slice_name), True)
        cb = load_and_prepare(os.path.join(cbct_folder, slice_name), True)
        preds = {lbl: load_and_prepare(os.path.join(folder, f"volume-{volume}", slice_name), False)
                 for lbl, folder in pred_folders.items()}
        # plot_profile(slice_name, x_coord, ct, cb, preds, volume)
        plot_profile_horizontal(slice_name, x_coord, ct, cb, preds, volume)
    if 'qq' in plot_types:
        plot_qq(volume, gt_folder, cbct_folder, pred_folders)
    if 'hist' in plot_types:
        plot_hi6t(volume, gt_folder, cbct_folder, pred_folders)
    if 'ba' in plot_types:
        plot_bland_altman(volume, gt_folder, cbct_folder, pred_folders)
    if 'ssim' in plot_types:
        plot_ssim(slice_name, gt_folder, cbct_folder, pred_folders, volume)
    if 'slice_metrics' in plot_types:
        plot_slice_metrics(volume, gt_folder, cbct_folder, pred_folders)

# Main
if __name__ == '__main__':
    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/490/test")
    pred_folders = {
        # 'v3': os.path.expanduser("~/thesis/predictions/predctions_controlnet_v3"),
        'v7': os.path.expanduser("~/thesis/predictions/predictions_controlnet_v7-data-augmentation"),
        # 'nl2': os.path.expanduser("~/thesis/predictions/predictions_tanh_v2")
    }
    volumes = VOLUMES
    current_vol = random.choice(volumes)
    preload_volume(current_vol, gt_folder, cbct_folder, pred_folders)
    print(f"[Main] Initial preload of volume {current_vol}")
    while True:
        next_vol = random.choice([v for v in volumes if v != current_vol])
        preload_volume(next_vol, gt_folder, cbct_folder, pred_folders)
        print(f"[Main] Preloading next volume {next_vol}")
        slices = list_slices(current_vol, gt_folder)
        if not slices:
            current_vol = next_vol
            continue
        slice_name = random.choice(slices)
        slice_name = 'volume-8_slice_107.npy'
        plot_multi(slice_name, x_coord=140, gt_folder=gt_folder,
                   cbct_folder=cbct_folder, pred_folders=pred_folders,
                   volume=current_vol)
        current_vol = next_vol
