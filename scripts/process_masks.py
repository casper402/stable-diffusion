#!/usr/bin/env python
import os
import itertools
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_and_split_mask(mask_path, ct_shape=None):
    """
    Loads a 4D mask (X,Y,2,Z), returns liver and tumor masks aligned to ct_shape.
    """
    data = nib.load(mask_path).get_fdata()
    data = np.squeeze(data)
    if data.ndim != 4 or data.shape[2] != 2:
        raise ValueError(f"Expected (X,Y,2,Z), got {data.shape}")

    liver = (data[:, :, 0, :] > 0)
    tumor = (data[:, :, 1, :] > 0)

    if ct_shape is not None and liver.shape != ct_shape:
        # Attempt permutations to match CT shape
        for name, arr in (('liver', liver), ('tumor', tumor)):
            if arr.shape != ct_shape:
                for perm in itertools.permutations(range(3)):
                    if arr.transpose(perm).shape == ct_shape:
                        if name == 'liver':
                            liver = arr.transpose(perm)
                        else:
                            tumor = arr.transpose(perm)
                        break
                else:
                    raise ValueError(f"Could not align {name} mask {arr.shape} to CT shape {ct_shape}")
    return liver.astype(bool), tumor.astype(bool)

def debug_visualize(ct_dir, mask_dir, volume_idx, ct_shape,
                    vmin_ct=-1000, vmax_ct=1000,
                    overlay_alpha=0.2,
                    threshold=30, max_images=10):
    """
    For a single volume, show up to max_images slices where tumor pixels > threshold.
    Plots: liver mask, tumor mask, CT only, and CT+overlay.
    """
    # Paths
    ct_path   = os.path.join(ct_dir,   f"volume-{volume_idx}.nii")
    mask_path = os.path.join(mask_dir, f"{volume_idx}.nii")
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        print(f"CT or mask for volume {volume_idx} not found.")
        return

    # Load CT volume and clip
    ct_vol = nib.load(ct_path).get_fdata(dtype=float)
    ct_vol = np.clip(ct_vol, vmin_ct, vmax_ct)

    # Load masks aligned to CT shape
    liver_mask, tumor_mask = load_and_split_mask(mask_path, ct_shape)
    z_dim = ct_shape[2]

    count = 0
    for z in range(z_dim):
        tu_slice = tumor_mask[:, :, z]
        num_pixels = int(np.sum(tu_slice))
        if num_pixels <= threshold:
            continue

        # Prepare slices (transpose so orientation matches display)
        lv = liver_mask[:, :, z].T
        tu = tumor_mask[:, :, z].T
        ct_slice = ct_vol[:, :, z].T

        # Plot 4 panels
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        ax0, ax1, ax2, ax3 = axes
        ax0.imshow(lv,  cmap="Reds",   origin="lower", vmin=0, vmax=1)
        ax0.set_title(f"Liver Mask\n(vol {volume_idx}, z={z})"); ax0.axis('off')

        ax1.imshow(tu,  cmap="Blues",  origin="lower", vmin=0, vmax=1)
        ax1.set_title(f"Tumor Mask (n={num_pixels})\n(vol {volume_idx}, z={z})"); ax1.axis('off')

        ax2.imshow(ct_slice, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct)
        ax2.set_title(f"CT Only\n(vol {volume_idx}, z={z})"); ax2.axis('off')

        ax3.imshow(ct_slice, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct)
        ax3.imshow(np.ma.masked_where(~lv, lv), cmap="Reds", alpha=overlay_alpha, origin="lower", vmin=0, vmax=1)
        ax3.imshow(np.ma.masked_where(~tu, tu), cmap="Blues", alpha=overlay_alpha, origin="lower", vmin=0, vmax=1)
        ax3.set_title(f"CT+Overlay (n={num_pixels})\n(vol {volume_idx}, z={z})"); ax3.axis('off')

        plt.tight_layout()
        plt.show()

        count += 1
        if count >= max_images:
            break


def process_one_volume(args):
    """
    Process and save mask slices aligned to CT shape.
    """
    mask_folder, mask_fname, out_liver, out_tumor, ct_shape = args
    idx = os.path.splitext(mask_fname)[0]
    mask_path = os.path.join(mask_folder, mask_fname)

    # Load aligned masks
    liver, tumor = load_and_split_mask(mask_path, ct_shape)
    num_slices = liver.shape[2]
    base = f"volume-{idx}" if not idx.startswith("volume-") else idx

    for i in range(num_slices):
        lv_slice = np.rot90(liver[:, :, i], k=-1)
        tu_slice = np.rot90(tumor[:, :, i], k=-1)
        slice_name = f"{base}_slice_{i:03d}.npy"
        np.save(os.path.join(out_liver, slice_name), lv_slice)
        np.save(os.path.join(out_tumor, slice_name), tu_slice)
    return idx

def process_all_parallel(mask_folder, out_liver, out_tumor,
                         min_vol=0, max_vol=130, ct_shape=None,
                         max_workers=None):
    tasks = []
    for idx in range(min_vol, max_vol + 1):
        fname = f"{idx}.nii"
        if not os.path.exists(os.path.join(mask_folder, fname)):
            continue
        tasks.append((mask_folder, fname, out_liver, out_tumor, ct_shape))

    print(f"Launching {len(tasks)} tasks with up to {max_workers or os.cpu_count()} workers...")
    os.makedirs(out_liver, exist_ok=True)
    os.makedirs(out_tumor, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_one_volume, task): task[1] for task in tasks}
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                vol_idx = fut.result()
                print(f"[OK ] Finished volume {vol_idx}")
            except Exception as e:
                print(f"[FAIL] {fname} → {e}")

if __name__ == "__main__":
    # ─── CONFIG ────────────────────────────────────────────────────────────
    ct_dir           = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT/"
    mask_dir         = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINMasksAlignedToCBCT/masks01/"
    output_liver     = "/Users/Niklas/thesis/training_data/masks/liver/"
    output_tumor     = "/Users/Niklas/thesis/training_data/masks/tumor/"
    min_volume_idx   = 0
    max_volume_idx   = 130
    max_workers      = None  # e.g. 8

    # Debug settings
    debug_volume_idx = 10    # which volume to inspect first
    threshold_pixels = 30    # tumor-pixel threshold
    max_debug_images = 10    # number of slices to show
    vmin_ct, vmax_ct = -1000, 1000
    overlay_alpha    = 0.2
    # ───────────────────────────────────────────────────────────────────────

    # Determine CT shape from debug volume
    sample_ct = os.path.join(ct_dir, f"volume-{debug_volume_idx}.nii")
    ct_shape = nib.load(sample_ct).get_fdata().shape

    # Step 1: debug visualize
    debug_visualize(
        ct_dir, mask_dir,
        volume_idx=debug_volume_idx,
        ct_shape=ct_shape,
        vmin_ct=vmin_ct, vmax_ct=vmax_ct,
        overlay_alpha=overlay_alpha,
        threshold=threshold_pixels,
        max_images=max_debug_images
    )

    # Step 2: process all volumes aligned to CT shape
    process_all_parallel(
        mask_dir,
        output_liver, output_tumor,
        min_vol=min_volume_idx,
        max_vol=max_volume_idx,
        ct_shape=ct_shape,
        max_workers=max_workers
    )
