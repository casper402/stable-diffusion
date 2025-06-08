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
#   (1) ─── SET A CONSISTENT, MINIMALIST STYLE ─────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("ggplot")

plt.rcParams["font.family"]    = "serif"
plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]      = 18
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["legend.fontsize"]= 16
plt.rcParams["xtick.labelsize"]= 12
plt.rcParams["ytick.labelsize"]= 12

# ────────────────────────────────────────────────────────────────────────────────
#   CONSTANTS AND “PIPELINE” FOR CT/CBCT: PADDING → RESIZE → CROP
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
    Pad a raw slice to 366×366 (fill=-1000), then resize to 256×256.
    Returns a NumPy array of shape 256×256.
    """
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    out = gt_transform(t)
    return out.squeeze(0).numpy()

def crop_back(arr: np.ndarray) -> np.ndarray:
    """
    Crop out ~45 px top & bottom from a 256×256 array → 166×256.
    """
    return arr[TOP_CROP:RES_H - BOTTOM_CROP, LEFT_CROP:RES_W - RIGHT_CROP]

def load_and_prepare(path: str, is_cbct: bool = True) -> np.ndarray:
    """
    Load a .npy file. If is_cbct, apply pad→resize→256. Then crop back to 166×256.
    """
    data = np.load(path)
    if is_cbct:
        data = apply_transform(data)
    return crop_back(data)

# ────────────────────────────────────────────────────────────────────────────────
#   SLICE SELECTION FOR EVALUATION
# ────────────────────────────────────────────────────────────────────────────────

SLICE_SELECT = {
    3: None, 8: (0, 354), 12: (0, 320), 26: None,
    32: (69, 269), 33: (59, 249), 35: (91, 268),
    54: (0, 330), 59: (0, 311), 61: (0, 315),
    106: None, 116: None, 129: (5, 346)
}
VOLUMES = list(SLICE_SELECT.keys())

def list_slices(volume: int, folder: str) -> list:
    """
    List slice filenames for a given volume, applying SLICE_SELECT filters.
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
#   WORKER FUNCTION FOR PARALLEL LOADING (TOP‐LEVEL)
# ────────────────────────────────────────────────────────────────────────────────

def _slice_worker_for_diff(args):
    """
    Top-level function to load one slice's CT, CBCT, and all predictions.
    Returns tuple of (ct_flat, cbct_flat, {label: pred_flat, …}).
    """
    volume, slice_name, gt_folder, cbct_folder, pred_folders = args

    # Paths for this slice:
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
#   FULL‐VOLUME LOADER THAT CALLS THE WORKER IN PARALLEL
# ────────────────────────────────────────────────────────────────────────────────

def _load_full_volume(volume, gt_folder, cbct_folder, pred_folders):
    """
    For a single volume:
      - Builds a list of slice names (applying SLICE_SELECT).
      - Uses ProcessPoolExecutor + _slice_worker_for_diff to load each slice.
      - Concatenates all flattened CT, CBCT, and pred arrays, returning them.
    """
    slices = list_slices(volume, gt_folder)
    ct_list, cbct_list = [], []
    preds_list = {lbl: [] for lbl in pred_folders}

    # Prepare arguments for the worker:
    args = [
        (volume, slice_name, gt_folder, cbct_folder, pred_folders)
        for slice_name in slices
    ]

    # Parallel map:
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
#   OVERLAYED HU‐DIFFERENCE HISTOGRAM (SINGLE PLOT)
# ────────────────────────────────────────────────────────────────────────────────

def plot_diff_hist_overlay(
    gt_folder: str,
    cbct_folder: str,
    pred_folders: dict,   # e.g. {"v7": "/path/to/v7_preds", …}
    volumes: list,        # e.g. VOLUMES
    bins: int = 100,
    save_path: str = None
):
    """
    1) Aggregate CT/CBCT/each‐pred over all `volumes`.
    2) Compute ΔHU = (CBCT – CT) and (pred – CT).
    3) Overlay both histograms—each as a filled, semi‐transparent area—on a SINGLE axes.
    """
    # ─── Step 1: Load & concatenate all volumes ─────────────────────────────────
    ct_accumulator   = []
    cbct_accumulator = []
    preds_accumulator = {lbl: [] for lbl in pred_folders}

    for v in volumes:
        print(f"[Overlay‐Hist] Loading volume {v} …")
        ct_all_v, cbct_all_v, preds_all_v = _load_full_volume(
            v, gt_folder, cbct_folder, pred_folders
        )
        ct_accumulator.append(ct_all_v)
        cbct_accumulator.append(cbct_all_v)
        for lbl in preds_all_v:
            preds_accumulator[lbl].append(preds_all_v[lbl])

    ct_all   = np.concatenate(ct_accumulator)
    cbct_all = np.concatenate(cbct_accumulator)
    preds_all = {lbl: np.concatenate(arrs) for lbl, arrs in preds_accumulator.items()}

    # ─── Step 2: Compute HU‐differences (1D arrays) ──────────────────────────────
    diffs = {
        "CBCT": cbct_all - ct_all
    }

    for lbl, arr in preds_all.items():
        diffs[lbl] = arr - ct_all

    filtered_diffs = {}
    for lbl, arr in diffs.items():
        # Keep only values in [−500, +500]
        filtered_diffs[lbl] = arr[np.abs(arr) <= 500]

    # Define bin edges from −500 to +500
    bin_edges = np.linspace(-500, 500, bins + 1)

    print("bin edges:", bin_edges)

    # ─── Step 3: Draw one overlaid histogram ────────────────────────────────────
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.10, 0.12, 0.88, 0.82])  # leave room at top for legend/title

    # Color choices:
    cbct_color = "#ff7f0e"    # orange
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, lbl in enumerate(diffs):
        arr = filtered_diffs[lbl]
        if lbl == "CBCT":
            color = cbct_color
        else:
            # Use muted blue for the first prediction; then cycle colors if you have more preds:
            color = "#d62728" if idx == 1 else default_cycle[(idx - 1) % len(default_cycle)]

        ax.hist(
            arr,
            bins=bin_edges,
            density=True,
            color=color,
            alpha=0.5,
            label=lbl
        )

    # ─── Remove top & right spines ─────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ─── Axis labels, title, tick formatting ──────────────────────────────────
    ax.set(
        xlabel="Δ HU",
        ylabel="Density",
    )
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.minorticks_off()

    # ─── Only horizontal grid lines ────────────────────────────────────────────
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(False)

    # ─── Legend above the plot, centered ───────────────────────────────────────
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(diffs),
        frameon=False,
        fontsize=16
    )

    ax.set_xlim(-300, 300)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"[Overlay‐Hist] Saved to: {save_path}")
    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
#   EXAMPLE CALL IN __main__
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/test")
    pred_folders = {
        "sCT": os.path.expanduser("~/thesis/predictions/predictions_controlnet_v7-data-augmentation"),
        # Add more predictions here if desired, e.g. "v3": "/path/to/v3_preds"
    }

    # Use all volumes in VOLUMES (defined above)
    out_pdf = os.path.expanduser("~/thesis/figures/hu_diff_hist_overlay_fixed.pdf")
    plot_diff_hist_overlay(
        gt_folder=gt_folder,
        cbct_folder=cbct_folder,
        pred_folders=pred_folders,
        volumes=VOLUMES,
        bins=200,
        save_path=out_pdf
    )
