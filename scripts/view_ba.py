#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from concurrent.futures import ProcessPoolExecutor
from matplotlib.ticker import MultipleLocator

# ────────────────────────────────────────────────────────────────────────────────
#   (1) ─── GLOBAL STYLE ────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("ggplot")

plt.rcParams["font.family"]    = "serif"
plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]      = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["legend.fontsize"]= 16
plt.rcParams["xtick.labelsize"]= 12
plt.rcParams["ytick.labelsize"]= 12

# ────────────────────────────────────────────────────────────────────────────────
#   (2) ─── CONSTANTS & TRANSFORMS FOR CT/CBCT PROCESSING
# ────────────────────────────────────────────────────────────────────────────────
DATA_RANGE = 2500.0    # HU range −1000…1000
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
#   (3) ─── SLICE SELECTION PER VOLUME
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
#   (4) ─── WORKER FOR PARALLEL LOADING (TOP‐LEVEL)
# ────────────────────────────────────────────────────────────────────────────────
def _slice_worker_for_diff(args):
    """
    Load one slice's CT, CBCT, and all predictions. Return flattened arrays.
    """
    volume, slice_name, gt_folder, cbct_folder, pred_folders = args

    ct_path   = os.path.join(gt_folder,   slice_name)
    cbct_path = os.path.join(cbct_folder, slice_name)

    ct_img   = load_and_prepare(ct_path,   is_cbct=True)
    cbct_img = load_and_prepare(cbct_path, is_cbct=True)

    preds_flat = {}
    for lbl, folder in pred_folders.items():
        pred_path = os.path.join(folder, f"volume-{volume}", slice_name)
        pred_arr  = load_and_prepare(pred_path, is_cbct=False)
        preds_flat[lbl] = pred_arr.flatten()

    return ct_img.flatten(), cbct_img.flatten(), preds_flat

# ────────────────────────────────────────────────────────────────────────────────
#   (5) ─── LOAD FULL VOLUME (USES WORKER ABOVE)
# ────────────────────────────────────────────────────────────────────────────────
def _load_full_volume(volume, gt_folder, cbct_folder, pred_folders):
    """
    For one volume, gather all slice‐wise CT, CBCT, and preds (flattened) and concatenate.
    Returns: (ct_all_v, cbct_all_v, preds_all_v_dict).
    """
    slices = list_slices(volume, gt_folder)
    ct_list, cbct_list = [], []
    preds_list = {lbl: [] for lbl in pred_folders}

    args = [
        (volume, slice_name, gt_folder, cbct_folder, pred_folders)
        for slice_name in slices
    ]

    with ProcessPoolExecutor() as executor:
        for ct_flat, cbct_flat, pred_dict in executor.map(_slice_worker_for_diff, args):
            ct_list.append(ct_flat)
            cbct_list.append(cbct_flat)
            for lbl, arr in pred_dict.items():
                preds_list[lbl].append(arr)

    ct_all_v   = np.concatenate(ct_list)
    cbct_all_v = np.concatenate(cbct_list)
    preds_all_v = {lbl: np.concatenate(arrs) for lbl, arrs in preds_list.items()}

    return ct_all_v, cbct_all_v, preds_all_v

# ────────────────────────────────────────────────────────────────────────────────
#   (6) ─── BLAND-ALTMAN SPLIT PLOT OVER ALL VOLUMES
#           (DIFF VS CT HU, TWO PANELS THAT FEEL CONNECTED)
# ────────────────────────────────────────────────────────────────────────────────

def plot_bland_altman_split_all(
    gt_folder: str,
    cbct_folder: str,
    pred_folders: dict,   # e.g. {"v7": "/path/to/v7_preds", …}
    volumes: list,        # list of volume indices (e.g. VOLUMES)
    save_path: str = None
):
    """
    1) Aggregate CT, CBCT, and each-prediction over all `volumes`.
    2) On the left panel: Bland–Altman (CBCT vs CT).
    3) On the right panel: Bland–Altman (all preds vs CT) overlaid.
    4) Panels share limits to feel “connected,” with a unified grid style.
    """

    BA_SUBSAMPLE = 3000   # Number of points to plot per target
    size = 10
    alpha = 0.5

    # ─── Step 1: Aggregate all volumes ───────────────────────────────────────────
    ct_accumulator   = []
    cbct_accumulator = []
    preds_accumulator = {lbl: [] for lbl in pred_folders}

    for v in volumes:
        print(f"[BA-SPLIT] Loading volume {v} …")
        ct_all_v, cbct_all_v, preds_all_v = _load_full_volume(
            v, gt_folder, cbct_folder, pred_folders
        )
        ct_accumulator.append(ct_all_v)
        cbct_accumulator.append(cbct_all_v)
        for lbl, arr in preds_all_v.items():
            preds_accumulator[lbl].append(arr)

    ct_all   = np.concatenate(ct_accumulator)
    cbct_all = np.concatenate(cbct_accumulator)
    preds_all = {lbl: np.concatenate(arrs) for lbl, arrs in preds_accumulator.items()}

    # ─── Step 2: Set up figure with two side-by-side panels ─────────────────────
    fig, (ax_cbct, ax_preds) = plt.subplots(
        1, 2,
        figsize=(16, 8),
        sharex=True,
        sharey=True
    )

    # Common color definitions
    cbct_color = "#ff7f0e"    # orange
    pred_blue  = "#1f77b4"    # muted blue
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ─── Step 3: Left panel — Bland–Altman CBCT vs CT ─────────────────────────
    diff_cbct = cbct_all - ct_all
    # Subsample for plotting
    cnt_cbct = ct_all.size
    if cnt_cbct > BA_SUBSAMPLE:
        sel_cbct = np.random.choice(cnt_cbct, BA_SUBSAMPLE, replace=False)
        m_ct_cbct, d_cbct = ct_all[sel_cbct], diff_cbct[sel_cbct]
    else:
        m_ct_cbct, d_cbct = ct_all, diff_cbct

    ax_cbct.scatter(
        m_ct_cbct,
        d_cbct,
        s=size,
        alpha=alpha,
        color=cbct_color,
        label="CBCT"
    )

    # Compute bias and limits (on full data)
    md_cbct = np.mean(diff_cbct)
    sd_cbct = np.std(diff_cbct)
    lo_cbct = md_cbct - 1.96 * sd_cbct
    hi_cbct = md_cbct + 1.96 * sd_cbct

    # Plot bias lines
    ax_cbct.axhline(md_cbct, linestyle='-', linewidth=1.5, color=cbct_color)
    ax_cbct.axhline(hi_cbct, linestyle='--', linewidth=1, color=cbct_color)
    ax_cbct.axhline(lo_cbct, linestyle='--', linewidth=1, color=cbct_color)

    # Annotate mean
    ax_cbct.text(
        1000, md_cbct,
        f"µ={md_cbct:.1f}",
        va='bottom',
        ha='right',
        color=cbct_color,
        fontsize=16
    )

    ax_cbct.set_title("CT vs CBCT")
    ax_cbct.set_xlabel("CT HU")
    ax_cbct.set_ylabel("ΔHU")
    ax_cbct.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # ─── Step 4: Right panel — Bland–Altman all preds vs CT ────────────────────
    for idx, (lbl, arr) in enumerate(preds_all.items()):
        diff_pred = arr - ct_all
        # Subsample for plotting
        cnt_pred = ct_all.size
        if cnt_pred > BA_SUBSAMPLE:
            sel_pred = np.random.choice(cnt_pred, BA_SUBSAMPLE, replace=False)
            m_ct_pred, d_pred = ct_all[sel_pred], diff_pred[sel_pred]
        else:
            m_ct_pred, d_pred = ct_all, diff_pred

        # Choose a color per prediction
        # color = pred_blue if idx == 0 else default_cycle[idx % len(default_cycle)]
        color = "#d62728"

        ax_preds.scatter(
            m_ct_pred,
            d_pred,
            s=size,
            alpha=alpha,
            color=color,
            label=lbl
        )

        # Compute bias & limits on full data
        md_pred = np.mean(diff_pred)
        sd_pred = np.std(diff_pred)
        lo_pred = md_pred - 1.96 * sd_pred
        hi_pred = md_pred + 1.96 * sd_pred

        # Plot bias lines
        ax_preds.axhline(md_pred, linestyle='-', linewidth=1.5, color=color)
        ax_preds.axhline(hi_pred, linestyle='--', linewidth=1, color=color)
        ax_preds.axhline(lo_pred, linestyle='--', linewidth=1, color=color)

        # Annotate mean
        ax_preds.text(
            1000, md_pred,
            f"µ={md_pred:.1f}",
            va='bottom',
            ha='right',
            color=color,
            fontsize=16
        )

    ax_preds.set_title("CT vs sCT")
    ax_preds.set_xlabel("CT HU")
    # oy-label is inherited from sharey; no need to set again
    ax_preds.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)

    # ─── Step 5: Enforce identical limits to keep panels “connected” ────────────
    # Determine common min/max for x and y
    all_diffs = np.concatenate([diff_cbct] + [arr - ct_all for arr in preds_all.values()])
    common_min = min(ct_all.min(), ct_all.min())
    common_max = max(ct_all.max(), ct_all.max())
    diff_min = all_diffs.min()
    diff_max = all_diffs.max()
    lim = max(abs(diff_min), abs(diff_max), abs(common_min), abs(common_max))
    ax_cbct.set_xlim(-1000, 1000)
    ax_cbct.set_ylim(-500, 500)
    ax_preds.set_xlim(-1000, 1000)
    ax_preds.set_ylim(-500, 500)

    # ─── Step 6: Finalize & save ─────────────────────────────────────────────────
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[BA-SPLIT] Saved to: {save_path}")
    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
#   (7) ─── MAIN: Stand‐alone execution
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/test")
    pred_folders = {
        "v7": os.path.expanduser(
            "~/thesis/predictions/predictions_controlnet_v7-data-augmentation"
        ),
        # Add more prediction folders if needed:
        # "v3": os.path.expanduser("~/thesis/predictions/predictions_controlnet_v3")
    }

    volumes = VOLUMES  # defined above via SLICE_SELECT

    # Optional: path to save PDF
    out_pdf = os.path.expanduser("~/thesis/figures/ba_split_diff_vs_ct_all_volumes.pdf")

    plot_bland_altman_split_all(
        gt_folder=gt_folder,
        cbct_folder=cbct_folder,
        pred_folders=pred_folders,
        volumes=volumes,
        save_path=out_pdf
    )
