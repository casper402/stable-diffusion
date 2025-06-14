#!/usr/bin/env python
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.ticker import MultipleLocator

# ────────────────────────────────────────────────────────────────────────────────
#   (1) ─── SET A CONSISTENT, MINIMALIST STYLE ─────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
    print("using seaborn-v0_8-paper style")
except OSError:
    print("falling back to ggplot style")
    plt.style.use("ggplot")

plt.rcParams["font.family"]    = "serif"
plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]      = 12
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"]= 16
plt.rcParams["xtick.labelsize"]= 14
plt.rcParams["ytick.labelsize"]= 10

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

ct_all           = df['ct_path'].tolist()
cbct_all         = df['cbct_490_path'].tolist()
ct_train         = df[df['split']=='train']['ct_path'].tolist()
ct_test          = df[df['split']=='test']['ct_path'].tolist()
cbct_train       = df[df['split']=='train']['cbct_490_path'].tolist()
cbct_test        = df[df['split']=='test']['cbct_490_path'].tolist()
ct_vol35         = df[df['ct_path'].str.contains('volume-35')]['ct_path'].tolist()
cbct_vol35       = df[df['cbct_490_path'].str.contains('volume-35')]['cbct_490_path'].tolist()

# ────────────────────────────────────────────────────────────────────────────────
#   PARAMETERS
# ────────────────────────────────────────────────────────────────────────────────
SLICE_LIMIT = None               # int or None\ nNUM_WORKERS = os.cpu_count() or 4
# DATA_FILE   = os.path.expanduser('~/thesis/hu_histograms.npz')
DATA_FILE   = os.path.expanduser('~/thesis/hu_histograms_train.npz')

# labels = [
#     'CT Train','CT Test','CBCT Train','CBCT Test','CT Vol-35','CBCT Vol-35'
# ]
labels = [
    'CT Train'
]

labels_CT = [
    'CT Train','CT Test','CT Vol-35'
]

labels_CBCT = [
    'CBCT Train','CBCT Test','CBCT Vol-35'
]

COLORS = {
    'CT Train':    '#1b9e77','CT Test':     '#d95f02',
    'CBCT Train':  '#1b9e77','CBCT Test':  '#d95f02',
    'CT Vol-35':   '#7570b3','CBCT Vol-35':'#7570b3'
}


STYLES = {
    'CT Train':    {'linestyle':'-','linewidth':3.0},
    'CT Test':     {'linestyle':'--','linewidth':3.0},
    'CBCT Train':  {'linestyle':'-','linewidth':3.0},
    'CBCT Test':   {'linestyle':'--','linewidth':3.0},
    'CT Vol-35':   {'linestyle':':','linewidth':2.5},
    'CBCT Vol-35': {'linestyle':':','linewidth':2.5},
}

# ────────────────────────────────────────────────────────────────────────────────
#   WORKER FUNCTION
# ────────────────────────────────────────────────────────────────────────────────
def _hist_for_file(fp: str, bins: np.ndarray, is_cbct: bool) -> np.ndarray:
    data = np.load(fp)
    if is_cbct:
        data = apply_transform(data)
    arr = crop_back(data)
    return np.histogram(arr.flatten(), bins=bins)[0]

# ────────────────────────────────────────────────────────────────────────────────
#   HISTOGRAM COMPUTATION
# ────────────────────────────────────────────────────────────────────────────────
def compute_histograms(groups, bins=np.linspace(-3000,10000,1000), limit=None, workers=1):
    results = {}
    centers = (bins[:-1] + bins[1:]) / 2
    for label, file_list, is_cbct in groups:
        files = file_list[:limit] if limit else file_list
        hist = np.zeros(len(bins)-1, dtype=np.float64)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_hist_for_file, fp, bins, is_cbct) for fp in files]
                for f in as_completed(futures):
                    hist += f.result()
        else:
            for fp in files:
                hist += _hist_for_file(fp, bins, is_cbct)
        results[label] = (centers, hist / hist.sum() if hist.sum()>0 else hist)
    return results

# ────────────────────────────────────────────────────────────────────────────────
#   SAVE HISTOGRAM DATA
# ────────────────────────────────────────────────────────────────────────────────
def save_histograms(hist_data, out_file=DATA_FILE):
    np.savez(out_file,
             **{f"{lbl}_x": data[0] for lbl,data in hist_data.items()},
             **{f"{lbl}_y": data[1] for lbl,data in hist_data.items()})
    print(f"Saved: {out_file}")

# ────────────────────────────────────────────────────────────────────────────────
#   LOAD & PLOT HISTOGRAM DATA WITH MATCHING STYLE
# ────────────────────────────────────────────────────────────────────────────────
def load_and_plot(npz_file=DATA_FILE):
    """
    Load centers and normalized counts from .npz and plot all curves
    using the thesis‐style profile‐plot settings.
    """

    print("*** Loading and plotting ***")

    data = np.load(npz_file)

    # Create figure & axes with same margins
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.10, 0.12, 0.88, 0.82])
    # Plot each curve
    for lbl in labels:
        x = data[f"{lbl}_x"]
        y = data[f"{lbl}_y"]
        ax.plot(x, y, label=lbl,
                color=COLORS[lbl], **STYLES[lbl])
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Axes limits
    ax.set_xlim(-3000, 10000)
    ax.set_ylim(0, 0.20)
    # Labels
    ax.set_xlabel('Hounsfield Units (HU)', fontsize=18)
    ax.set_ylabel('Normalized Frequency', fontsize=18)
    # Ticks
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.minorticks_off()
    # Grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(False)
    # Legend above plot
    # ax.legend(
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, 0.97),
    #     ncol=len(labels),
    #     frameon=False,
    #     fontsize=16
    # )
    save_path = os.path.expanduser("~/thesis/figures/hu_train_unclipped.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    print("SaVkkved histogram figure to:", save_path)
    plt.show()

def load_and_plot_split(npz_file=DATA_FILE):
    """
    Load centers and normalized counts from .npz and plot CT and CBCT histograms
    side by side with a shared y-axis.
    """
    # Load data
    data = np.load(npz_file)

    # Prepare figure and axes
    fig, (ax_ct, ax_cbct) = plt.subplots(
        1, 2,
        sharey=True,
        figsize=(12, 6),
        gridspec_kw={'width_ratios': [1, 1]}
    )

    # Plot CT curves on left
    for lbl in labels_CT:
        x = data[f"{lbl}_x"]
        y = data[f"{lbl}_y"]
        ax_ct.plot(x, y, label=lbl, color=COLORS[lbl], **STYLES[lbl])
    ax_ct.set_xlim(-1050, 1050)
    ax_ct.set_ylim(0, 0.16)
    ax_ct.set_xlabel('Hounsfield Units (HU)', fontsize=18)
    ax_ct.set_ylabel('Normalized Frequency', fontsize=18)
    ax_ct.set_title('CT Distribution')
    ax_ct.xaxis.set_major_locator(MultipleLocator(500))
    ax_ct.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_ct.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax_ct.spines['top'].set_visible(True)
    ax_ct.spines['right'].set_visible(True)
    ax_ct.legend(frameon=True, fontsize=14)

    # Plot CBCT curves on right
    for lbl in labels_CBCT:
        x = data[f"{lbl}_x"]
        y = data[f"{lbl}_y"]
        ax_cbct.plot(x, y, label=lbl, color=COLORS[lbl], **STYLES[lbl])
    ax_cbct.set_xlim(-1050, 1050)
    ax_cbct.set_xlabel('Hounsfield Units (HU)', fontsize=18)
    ax_cbct.set_title('CBCT Distribution')
    ax_cbct.xaxis.set_major_locator(MultipleLocator(500))
    ax_cbct.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
    ax_cbct.spines['top'].set_visible(True)
    ax_cbct.spines['right'].set_visible(True)
    ax_cbct.legend(frameon=True, fontsize=14)

    plt.tight_layout()

    save_path = os.path.expanduser("~/thesis/figures/hu_train_test.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    print("SaVkkved histogram figure to:", save_path)

    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
#   MAIN
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Compute & save
    groups = [
        ('CT Train', ct_train, True), ('CT Test', ct_test, True),
        # ('CBCT Train', cbct_train, True), ('CBCT Test', cbct_test, True),
        # ('CT Vol-35', ct_vol35, True), ('CBCT Vol-35', cbct_vol35, True)
    ]
    # hist_data = compute_histograms(groups, limit=SLICE_LIMIT, workers=NUM_WORKERS)
    # save_histograms(hist_data)

    # To plot after saving:
    load_and_plot()
