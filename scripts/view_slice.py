import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random

from evaluation import compute_mae, compute_rmse, compute_psnr, DATA_RANGE, ssim

SLICE_RANGES = {
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
    129: (5, 346),
}

# ──────── constants ───────────────────────────────────────────────────────────
DATA_RANGE = 2000.0    # CT range -1000…1000

ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B  # = 238 + 64 + 64 = 366
_pad_w = ORIG_W + PAD_L + PAD_R  # = 366 + 0 + 0   = 366
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))     # ≈ 45
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))     # ≈ 45
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))     # = 0
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))     # = 0

transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W)),
])

def apply_transform(np_img):
    """Convert NumPy → PIL → transformed PIL → NumPy"""
    pil = Image.fromarray(np_img)
    out = transform(pil)
    return np.array(out)

def crop_back(arr):
    """Crop transformed (256×256) image back to original-ish (166×256) size."""
    return arr[
        TOP_CROP:    RES_H - BOTTOM_CROP,   # = 256 - 45 = 211; so 45:211 → 166 px tall
        LEFT_CROP:   RES_W - RIGHT_CROP     # = 256 - 0  = 256; so 0:256 → 256 px wide
    ]

def load_processed_volume(volume_dir, volume_idx, needs_transform=False):
    """
    Loads all axial slices for a given volume index from volume_dir,
    optionally applies the same transform+crop to each slice,
    and stacks them into a 3D NumPy array of shape (Z, H, W).
    """
    pattern = os.path.join(volume_dir, f"volume-{volume_idx}_slice_*.npy")
    slice_files = sorted(glob.glob(pattern))
    if len(slice_files) == 0:
        raise FileNotFoundError(f"No slices found for volume {volume_idx} in {volume_dir}.")

    processed_slices = []
    for slc_path in slice_files:
        slc = np.load(slc_path)             # raw 2D slice (typically 238×366)
        if needs_transform:
            slc = apply_transform(slc)      # → (256×256) with padding + resize
        slc = crop_back(slc)                # → (166×256)
        processed_slices.append(slc)

    volume = np.stack(processed_slices, axis=0)  # shape: (Z, 166, 256)

    print(f"Loaded volume {volume_idx} from {volume_dir}  →  shape: {volume.shape}")
    return volume

def extract_slice(volume, orientation, idx):
    """
    Given a 3D volume array of shape (Z, H, W), return a 2D slice according to:
      - orientation == 'axial':    volume[idx,    :, :]
      - orientation == 'coronal':  volume[:, idx,    :]
      - orientation == 'sagittal': volume[:,    :, idx]
    """
    if orientation == "axial":
        return np.fliplr(volume[idx, :, :])
    elif orientation == "coronal":
        return np.fliplr(np.flipud(volume[:, idx, :]))
    elif orientation == "sagittal":
        return np.flipud(volume[:, :, idx])
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

def resize_to_256(slc):
    """
    Take any 2D NumPy array 'slc' (shape = (h0, w0)) and
    resize it to exactly (256, 256) via PIL bilinear.
    """
    pil = Image.fromarray(slc.astype(np.int16))
    resized = pil.resize((256, 256), resample=Image.BILINEAR)
    return np.array(resized)

fontsize = 12

def plot_combined_oriented_slices(
    volume_idx,
    gt_info,        # tuple: (gt_dir, gt_display_name, gt_needs_transform)
    test_dirs_info  # list of tuples: [(dir_path, display_name, needs_transform), ...]
):
    """
    Loads the entire 3D volume (GT + each test directory) for volume_idx,
    selects one axial/coronal/sagittal slice, resizes them to (256×256),
    then plots them in a single 5×N figure:
      • Row 0: Axial (window = [-1000, +1000])
      • Row 1: Coronal (window = [-1000, +1000])
      • Row 2: Sagittal (window = [-1000, +1000])
      • Row 3: Axial (window = [-1000, +300])
      • Row 4: Axial (window = [ -400, +400])

    The column headers (GT + each test) appear only once at the top row.
    Each row has a bold, left‐marginal label.
    """
    # 1) Unpack GT info
    gt_dir, gt_name, gt_needs_trans = gt_info

    # 2) Load GT volume
    print(f"\n--- Loading GT volume {volume_idx} from: {gt_dir} (transform={gt_needs_trans}) ---")
    gt_vol = load_processed_volume(gt_dir, volume_idx, needs_transform=gt_needs_trans)

    # 3) Load each test volume
    test_vols = []
    test_names = []
    for (tdir, tname, tneed) in test_dirs_info:
        print(f"--- Loading TEST volume from: {tdir} (transform={tneed}) ---")
        vol = load_processed_volume(tdir, volume_idx, needs_transform=tneed)
        test_vols.append(vol)
        test_names.append(tname)

    # 4) Combine GT + tests
    all_vols = [gt_vol] + test_vols
    all_names = [gt_name] + test_names

    Z, H, W = gt_vol.shape
    print(f"\nGT volume dims: Z={Z}, H={H}, W={W}.  All slices will be resized to (256×256).\n")

    # 5) Determine valid axial slice range (if in SLICE_RANGES)
    if (volume_idx in SLICE_RANGES) and (SLICE_RANGES[volume_idx] is not None):
        lb, ub = SLICE_RANGES[volume_idx]
        ub = min(ub, Z - 1)
    else:
        lb, ub = 0, Z - 1

    # 6) Pick one index for axial, and one each for coronal/sagittal
    axial_idx    = random.randint(lb, ub)
    coronal_idx  = random.randint(0, H - 1)
    sagittal_idx = random.randint(0, W - 1)
    # (For reproducibility, you can hardcode: axial_idx, coronal_idx, sagittal_idx = 150, 150, 60)
    axial_idx, coronal_idx, sagittal_idx = 150, 150, 60

    print(f"Chosen slice indices → Axial: {axial_idx}, Coronal: {coronal_idx}, Sagittal: {sagittal_idx}\n")

    # 7) Extract & resize every needed slice for GT + tests
    #    • Axial-default (–1000→+1000)
    raw_axial   = [extract_slice(vol, "axial",    axial_idx)   for vol in all_vols]
    resized_axial = [resize_to_256(slc) for slc in raw_axial]

    #    • Coronal (–1000→+1000)
    raw_coronal   = [extract_slice(vol, "coronal", coronal_idx)   for vol in all_vols]
    resized_coronal = [resize_to_256(slc) for slc in raw_coronal]

    #    • Sagittal (–1000→+1000)
    raw_sagittal   = [extract_slice(vol, "sagittal", sagittal_idx) for vol in all_vols]
    resized_sagittal = [resize_to_256(slc) for slc in raw_sagittal]

    # ─────────────────── Font + Style Tweaks ───────────────────
    global fontsize
    plt.rcParams["font.family"]    = "serif"
    plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
    plt.rcParams["font.size"]      = fontsize
    plt.rcParams["axes.titlesize"] = fontsize
    plt.rcParams["axes.labelsize"] = fontsize
    fontsize += 2

    # 8) Create a 5×N figure (5 rows: Axial,Coronal,Sagittal,Axial[−1000,300],Axial[−400,400])
    row_data = [
        # (list_of_resized_slices,   vmin,   vmax,       row_label)
        (resized_axial,              -1000,  1000,      "Axial"),
        (resized_axial,              -400,  400,       "Axial"),
        (resized_coronal,            -400,  400,      "Coronal"),
        (resized_sagittal,           -400,  400,      "Sagittal"),
    ]

    n_rows = len(row_data)
    n_cols = len(all_vols)  # GT + each test
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
        constrained_layout=True
    )
    # In case n_cols=1, axes will be shape (5,), so we ensure 2D indexing
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)


    for row_idx, (slice_list, vmin_val, vmax_val, row_label) in enumerate(row_data):
        # Print metrics only for the “comparison” rows (rows 0, 1, 2).
        if row_idx < 3:
            print(f"\n--- {row_label}, slice index = "
                  f"{[axial_idx, coronal_idx, sagittal_idx][row_idx]} ---")
            gt_slice = slice_list[0]
            for col_idx, name in enumerate(all_names[1:], start=1):
                test_slice = slice_list[col_idx]
                mae_val  = compute_mae(test_slice, gt_slice)
                rmse_val = compute_rmse(test_slice, gt_slice)
                psnr_val = compute_psnr(test_slice, gt_slice, data_range=DATA_RANGE)
                ssim_val = ssim(gt_slice, test_slice, data_range=DATA_RANGE)
                print(f"{name:20s}  →  MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, "
                      f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.2f}")

        # Plotting each column in this row
        for col_idx, slc256 in enumerate(slice_list):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc256, cmap="gray", vmin=vmin_val, vmax=vmax_val)
            ax.axis("off")

            # Column titles only on the top row:
            if row_idx == 0:
                ax.set_title(all_names[col_idx], pad=6)

            # Row label on leftmost column (once per row)
            if col_idx == 0:
                ax.text(
                    -0.06, 0.5, row_label,
                    va="center", ha="center",
                    rotation="vertical",
                    transform=ax.transAxes
                )

    # 10) Super‐title
    # fig.suptitle(
    #     f"Volume {volume_idx}  —  Axial/Coronal/Sagittal + 2×Axial‐Windows",
    #     fontsize=16,
    #     fontweight="bold"
    # )

    # 11) Show the combined figure
    print("fontsize:", fontsize)
    plt.show()
    output_path = os.path.expanduser("/Users/Niklas/thesis/figures/comparison.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print("saved to:", output_path)



if __name__ == "__main__":
    volume_idx = 3

    # ─────────────────── Example usage ───────────────────
    # Ground‐truth directory info: (path, custom_name, needs_transform_flag)
    gt_info = (
        os.path.expanduser("/Users/Niklas/thesis/training_data/CT/test"),
        "CT",
        True   # ← set to True/False depending on whether GT needs padding+resize
    )

    # List of test directories, each as (path, custom_name, needs_transform_flag)
    test_dirs_info = [
        (
            os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/256/test"),
            "CBCT",
            True
        ),
        (
            os.path.expanduser(f"/Users/Niklas/thesis/predictions/thesis-ready/256/best-model/50-steps-linear/volume-{volume_idx}"),
            "sCT (ours)",
            False
        ),
        (
            os.path.expanduser(
                "/Users/Niklas/thesis/predictions/predictions_tanh_v5/volume-3"
            ),
            "sCT (CycleGan)",
            False
        ),
    ]
    

    try:
        while True:
            plot_combined_oriented_slices(
                volume_idx=volume_idx,
                gt_info=gt_info,
                test_dirs_info=test_dirs_info
            )
    except KeyboardInterrupt:
        print("\nStopped by user.")
