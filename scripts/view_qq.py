#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from concurrent.futures import ProcessPoolExecutor
from matplotlib.patches import Rectangle, ConnectionPatch
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
#   (6) ─── OVERLAYED Q-Q PLOT WITH BOTTOM‐RIGHT ZOOMED INSET
#           (WITH RECOMPUTED QUANTILES FOR INSET)
# ────────────────────────────────────────────────────────────────────────────────
QQ_QUANTILES = 500      # Number of quantile points for main plot
QQ_QUANTILES_ZOOM = 100 # Number of quantile points within zoom range

def plot_qq_with_inset(
    gt_folder: str,
    cbct_folder: str,
    pred_folders: dict,   # e.g. {"v7": "/path/to/v7_preds", …}
    volumes: list,        # list of volume indices (e.g. VOLUMES)
    zoom_range: tuple = (-150, 150),  # (min, max) for both axes
    save_path: str = None
):
    """
    1) Aggregate CT, CBCT, and each-prediction over all `volumes`.
    2) Compute QQ quantiles for CT vs CBCT and CT vs each-pred (main quantiles).
    3) Also compute QQ quantiles restricted to CT/CBCT/pred arrays within zoom_range.
    4) Plot a big QQ plot with all quantiles, plus a bottom-right inset that
       shows the zoomed-in quantiles (recomputed) for easier detail.
    5) Draw a dashed rectangle on the big plot around zoom_range and connect
       to the inset with slanted gray lines.
    """
    # ─── Step 1: Aggregate all volumes ───────────────────────────────────────────
    ct_accumulator   = []
    cbct_accumulator = []
    preds_accumulator = {lbl: [] for lbl in pred_folders}

    for v in volumes:
        print(f"[QQ‐Zoom] Loading volume {v} …")
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

    # ─── Step 2: Compute main quantiles ──────────────────────────────────────────
    qs_main = np.linspace(0, 1, QQ_QUANTILES)
    ct_q_main = np.quantile(ct_all, qs_main)

    targets = {"CBCT": cbct_all}
    for lbl, arr in preds_all.items():
        targets[lbl] = arr

    target_quantiles_main = {
        lbl: np.quantile(arr, qs_main)
        for lbl, arr in targets.items()
    }

    # ─── Step 3: Compute zoomed quantiles (only for values in zoom_range) ───────
    x_min, x_max = zoom_range
    y_min, y_max = zoom_range

    # Filter each array to its own zoomed values
    ct_zoom = ct_all[(ct_all >= x_min) & (ct_all <= x_max)]
    cbct_zoom = cbct_all[(cbct_all >= y_min) & (cbct_all <= y_max)]
    preds_zoom = {
        lbl: arr[(arr >= y_min) & (arr <= y_max)]
        for lbl, arr in preds_all.items()
    }

    # Compute quantiles for zoomed arrays
    qs_zoom = np.linspace(0, 1, QQ_QUANTILES_ZOOM)

    if ct_zoom.size > 0:
        ct_q_zoom = np.quantile(ct_zoom, qs_zoom)
    else:
        ct_q_zoom = np.array([])

    target_quantiles_zoom = {}
    target_quantiles_zoom["CBCT"] = (
        np.quantile(cbct_zoom, qs_zoom) if cbct_zoom.size > 0 else np.array([])
    )
    for lbl, arr_zoom in preds_zoom.items():
        target_quantiles_zoom[lbl] = (
            np.quantile(arr_zoom, qs_zoom) if arr_zoom.size > 0 else np.array([])
        )

    # ─── Step 4: Plot main QQ axes ──────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 8))
    ax_main = fig.add_axes([0.12, 0.12, 0.85, 0.85])  # [left, bottom, width, height]

    # Identity line (dashed)
    mn_main, mx_main = ct_q_main.min(), ct_q_main.max()
    ax_main.plot(
        [mn_main, mx_main],
        [mn_main, mx_main],
        color="k",
        linestyle="--",
        linewidth=1.5,
        label="Identity"
    )

    # Colors
    cbct_color = "#ff7f0e"    # orange
    pred_blue  = "#1f77b4"    # muted blue
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot CBCT vs CT (main)
    ax_main.plot(
        ct_q_main,
        target_quantiles_main["CBCT"],
        marker="o",
        markersize=3,
        linestyle="",
        color=cbct_color,
        alpha=0.7,
        label="CBCT"
    )

    # Plot each prediction vs CT (main)
    idx = 1
    for lbl, q_arr in target_quantiles_main.items():
        if lbl == "CBCT":
            continue
        color = pred_blue if idx == 1 else default_cycle[(idx - 2) % len(default_cycle)]
        ax_main.plot(
            ct_q_main,
            q_arr,
            marker="o",
            markersize=3,
            linestyle="",
            color="#d62728",
            alpha=0.7,
            label="sCT"
        )
        idx += 1

    # Style main plot
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.set(
        xlabel="CT Quantiles",
        ylabel="Target Quantiles",
    )
    ax_main.xaxis.set_major_locator(MultipleLocator(500))
    ax_main.yaxis.set_major_locator(MultipleLocator(500))
    ax_main.minorticks_off()
    ax_main.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax_main.xaxis.grid(False)
    ax_main.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(target_quantiles_main) + 1,  # +1 for Identity
        frameon=False,
        fontsize=16,
        markerscale=2.0
    )

    # ─── Step 5: Draw dashed rectangle on main plot around zoom_range ───────────
    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=1,
        edgecolor="gray",
        linestyle="--",
        fill=False,
        transform=ax_main.transData
    )
    ax_main.add_patch(rect)

    # ─── Step 6: Add bottom-right inset ─────────────────────────────────────────
    ax_inset = fig.add_axes([0.65, 0.17, 0.30, 0.30])  # [left, bottom, width, height]

    # Identity line in inset
    ax_inset.plot(
        [x_min, x_max],
        [x_min, x_max],
        color="k",
        linestyle="--",
        linewidth=1.0
    )

    # Plot CBCT zoom quantiles if available
    if ct_q_zoom.size > 0 and target_quantiles_zoom["CBCT"].size > 0:
        ax_inset.plot(
            ct_q_zoom,
            target_quantiles_zoom["CBCT"],
            marker="o",
            markersize=3,
            linestyle="",
            color=cbct_color,
            alpha=0.7
        )

    # Plot each prediction zoom quantiles if available
    idx = 1
    for lbl, q_arr_zoom in target_quantiles_zoom.items():
        if lbl == "CBCT":
            continue
        color = pred_blue if idx == 1 else default_cycle[(idx - 2) % len(default_cycle)]
        if ct_q_zoom.size > 0 and q_arr_zoom.size > 0:
            ax_inset.plot(
                ct_q_zoom,
                q_arr_zoom,
                marker="o",
                markersize=3,
                linestyle="",
                color="#d62728",
                alpha=0.7
            )
        idx += 1

    # Style inset
    ax_inset.set_xlim(x_min, x_max)
    ax_inset.set_ylim(y_min, y_max)
    # ax_inset.spines["top"].set_visible(False)
    # ax_inset.spines["right"].set_visible(False)
    ax_inset.xaxis.set_major_locator(MultipleLocator(50))
    ax_inset.yaxis.set_major_locator(MultipleLocator(50))
    ax_inset.minorticks_off()
    ax_inset.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax_inset.xaxis.grid(False)
    # Now style all four spines the same way:
    for side in ['left', 'right', 'top', 'bottom']:
        spine = ax_inset.spines[side]
        spine.set_linestyle('--') 
        spine.set_linewidth(1)
        spine.set_color('gray')

    # ─── Step 6.1: Draw dashed arrow from rectangle’s right-middle → inset top-center ─────
    #  a) midpoint of the right side of the rectangle (in data coordinates of ax_main)
    mid_right = (x_max, (y_min + y_max) / 2)

    #  b) top-center of the inset (in AxesFraction coordinates of ax_inset)
    top_mid_inset = (0.5, 1.0)

    #  c) create a dashed arrow from A → B
    con = ConnectionPatch(
        xyA=mid_right,             # A: right‐middle point of rect, in data coords
        coordsA=ax_main.transData, # tell it “xyA is in ax_main’s data space”
        xyB=top_mid_inset,         # B: top‐center of inset, in axes‐fraction
        coordsB=ax_inset.transAxes,# tell it “xyB is in ax_inset’s axes space”
        arrowstyle="->",           # arrowhead at the end
        linestyle="-",
        color="gray",
        linewidth=1.0,
    )
    ax_inset.add_artist(con)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[QQ‐Zoom] Saved to: {save_path}")
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
    out_pdf = os.path.expanduser("~/thesis/figures/qq_overlay_with_zoom.pdf")

    plot_qq_with_inset(
        gt_folder=gt_folder,
        cbct_folder=cbct_folder,
        pred_folders=pred_folders,
        volumes=volumes,
        zoom_range=(-150, 150),
        save_path=out_pdf
    )
