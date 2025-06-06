#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim  # imported but not used here
from skimage.measure import profile_line  # imported but not used here
from matplotlib.ticker import MultipleLocator

# ────────────────────────────────────────────────────────────────────────────────
#   (1) ─── GLOBAL STYLE (from the profile-plot example) ────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("ggplot")

plt.rcParams["font.family"]     = "serif"
plt.rcParams["font.serif"]      = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]       = 12
plt.rcParams["axes.titlesize"]  = 14
plt.rcParams["axes.labelsize"]  = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# ────────────────────────────────────────────────────────────────────────────────
#   (2) ─── CONSTANTS & TRANSFORMS FOR CT/CBCT PROCESSING (unchanged) ────────────
# ────────────────────────────────────────────────────────────────────────────────
DATA_RANGE = 2000.0    # HU range −1000…1000
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))    # ≈45
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))    # ≈45
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))    # 0
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))    # 0

gt_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W), interpolation=InterpolationMode.BILINEAR),
])

def apply_transform(img_np: np.ndarray) -> np.ndarray:
    """
    Pad a raw slice to 366×366 (fill = -1000), then resize to 256×256.
    """
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    out = gt_transform(t)  # Tensor shape (1×256×256)
    return out.squeeze(0).numpy()

def crop_back(arr: np.ndarray) -> np.ndarray:
    """
    Crop the padded+resized array (256×256) down to the central 166×256 region.
    """
    return arr[TOP_CROP:RES_H - BOTTOM_CROP, LEFT_CROP:RES_W - RIGHT_CROP]

def load_and_prepare(path: str, is_cbct: bool = True) -> np.ndarray:
    """
    Load a .npy file, optionally apply pad→resize if CBCT, then crop back to 166×256.
    """
    data = np.load(path)
    if is_cbct:
        data = apply_transform(data)
    return crop_back(data)

# ────────────────────────────────────────────────────────────────────────────────
#   (3) ─── SLICE SELECTION PER VOLUME (unchanged) ───────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
SLICE_SELECT = {
    3: None,
    8: (0, 354),
    12: (0, 320),
    26: None,
    32: (69, 269),
    33: (59, 249),
    35: (91, 268),
    54: (0, 330),
    59: (0, 311),
    61: (0, 315),
    106: None,
    116: None,
    129: (5, 346)
}
VOLUMES = list(SLICE_SELECT.keys())
VOLUMES = [3]

def list_slices(volume: int, folder: str) -> list:
    """
    List slice filenames for a given volume, filtering by SLICE_SELECT.
    """
    pattern = os.path.join(folder, f"volume-{volume}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    selector = SLICE_SELECT.get(volume)
    if selector is None:
        return [os.path.basename(f) for f in files]
    if isinstance(selector, tuple) and len(selector) == 2:
        start, end = selector
        return [
            os.path.basename(f)
            for f in files
            if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end
        ]
    if isinstance(selector, list):
        valid = set(selector)
        return [
            os.path.basename(f)
            for f in files
            if int(os.path.basename(f).split('_')[-1].split('.')[0]) in valid
        ]
    return [os.path.basename(f) for f in files]

# ────────────────────────────────────────────────────────────────────────────────
#   (4) ─── WORKER FOR PARALLEL LOADING OF MAE (SLICE‐WISE) ──────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def _slice_worker_for_mae(args):
    """
    Load one slice's CT, CBCT, and all predictions. Return MAE values.
    """
    volume, slice_name, gt_folder, cbct_folder, pred_folders = args

    ct_img   = load_and_prepare(os.path.join(gt_folder, slice_name), is_cbct=True)
    cbct_img = load_and_prepare(os.path.join(cbct_folder, slice_name), is_cbct=True)

    mae_cbct = np.mean(np.abs(cbct_img - ct_img))
    mae_preds = {}
    for lbl, folder in pred_folders.items():
        pred_img = load_and_prepare(os.path.join(folder, f"volume-{volume}", slice_name), is_cbct=False)
        mae_preds[lbl] = np.mean(np.abs(pred_img - ct_img))

    return mae_cbct, mae_preds

def _load_all_slice_mae(volume, gt_folder, cbct_folder, pred_folders):
    """
    For one volume, gather per‐slice MAE for CBCT and each prediction.
    Returns: (mae_cbct_array, mae_preds_dict_of_arrays, slice_count)
    """
    slice_names = list_slices(volume, gt_folder)
    mae_cbct_list = []
    mae_preds_lists = {lbl: [] for lbl in pred_folders}

    args = [
        (volume, slice_name, gt_folder, cbct_folder, pred_folders)
        for slice_name in slice_names
    ]

    with ProcessPoolExecutor() as executor:
        for mae_cbct, mae_preds in executor.map(_slice_worker_for_mae, args):
            mae_cbct_list.append(mae_cbct)
            for lbl, val in mae_preds.items():
                mae_preds_lists[lbl].append(val)

    return (
        np.array(mae_cbct_list),
        {lbl: np.array(vals) for lbl, vals in mae_preds_lists.items()},
        len(slice_names)
    )

# ────────────────────────────────────────────────────────────────────────────────
#   (5) ─── REDESIGNED SLICE‐WISE MAE PLOT (ONLY MAE ROW, PROFILE STYLE) ───────
# ────────────────────────────────────────────────────────────────────────────────
def plot_slice_metrics_mae(
    volume: int,
    gt_folder: str,
    cbct_folder: str,
    pred_folders: dict,
    save_path: str = None
):
    """
    Compute slice‐wise MAE (CT vs CBCT and CT vs each prediction) for one volume,
    then plot all MAE curves together in a single “MAE vs slice‐index” panel
    using the minimalist, profile-style formatting.
    """
    # ─── Load per-slice MAE ─────────────────────────────────────────────────────
    mae_cbct_arr, mae_preds_dict, slice_count = _load_all_slice_mae(
        volume, gt_folder, cbct_folder, pred_folders
    )
    x = np.arange(slice_count)

    # ─── Color palette & line styles (similar to profile plot) ─────────────────
    COLORS = {"CBCT": "#ff7f0e"}  # orange
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, lbl in enumerate(pred_folders.keys()):
        COLORS[lbl] = cycle_colors[idx % len(cycle_colors)]

    # ─── Create figure & axis ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot CBCT MAE
    ax.plot(
        x,
        mae_cbct_arr,
        label="CBCT",
        color=COLORS["CBCT"],
        linestyle="-",
        linewidth=2.5,
        alpha=0.8
    )

    # Plot each prediction MAE
    for lbl, vals in mae_preds_dict.items():
        ax.plot(
            x,
            vals,
            label=lbl,
            color="#d62728",
            linestyle="-",
            linewidth=2.5,
            alpha=0.8
        )

    # ─── Remove top & right spines for a cleaner look ─────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ─── Axes labels & title ───────────────────────────────────────────────────
    ax.set(
        xlabel="Slice index",
        ylabel="MAE (HU)",
    )

    # ─── Tick formatting ────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(MultipleLocator(50))   # major tick every 10 slices
    ax.yaxis.set_major_locator(MultipleLocator(20))   # major tick every 20 HU
    ax.minorticks_off()

    # ─── Grid: only horizontal lines, slightly bolder & less transparent ──────
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlim(0, 364)
    ax.set_xlim(-0.5, 364)

    # ─── Legend above the plot ─────────────────────────────────────────────────
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(pred_folders) + 1,
        frameon=False
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[MAE] Saved to: {save_path}")
    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
#   (6) ─── MAIN: Stand‐alone execution (Loop over all volumes) ────────────────
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/test")
    pred_folders = {
        "sCT": os.path.expanduser(
            "~/thesis/predictions/predictions_controlnet_v7-data-augmentation"
        ),
        # "v3": os.path.expanduser("~/thesis/predictions/predictions_controlnet_v3"),
    }

    for vol in VOLUMES:
        print(f"[Main] Plotting slice-wise MAE for volume {vol}")
        out_pdf = os.path.expanduser(f"~/thesis/figures/slice_mae_volume_{vol}.pdf")
        plot_slice_metrics_mae(
            volume=vol,
            gt_folder=gt_folder,
            cbct_folder=cbct_folder,
            pred_folders=pred_folders,
            save_path=out_pdf
        )
