#!/usr/bin/env python3
"""
Postprocess CBCT-to-sCT output back to original geometry,
with everything outside the ROI set to -1000.

1. Load per-slice .npy predictions (256×256), reverse-pad+reverse-crop them back to the 340×238 ROI.
2. Insert each ROI back into a constant -1000 512×512×z volume.
3. Save as a NIfTI, preserving original affine/header.
"""

import os
import glob
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

# --- User settings -----------------------------------------------------------
ORDER = 1  # nearest‐neighbor
input_dir = os.path.expanduser(
    "/Users/Niklas/thesis/predictions/predictions_clinicV2_cropped"
)
original_nifti = os.path.expanduser(
    "/Users/Niklas/thesis/training_data/Casper-klinik/CBCT.nii.gz"
)
output_filename = os.path.expanduser(
    f"/Users/Niklas/thesis/training_data/Casper-klinik/sCT-cropped-steps-{ORDER}.nii.gz"
)

# ROI/CROP/PAD parameters (must match your training‐time transform)
x0, x1 = 85, 425           # crop X-range on 512×512
y0, y1 = 134, 372          # crop Y-range on 512×512
pad_left, pad_top = 13, 64
pad_right, pad_bottom = 13, 64

crop_w = x1 - x0           # 340
crop_h = y1 - y0           # 238
padded_size = crop_w + pad_left + pad_right   # 366
zoom_factor = padded_size / 256               # ≈1.4297

# --- Load original geometry info --------------------------------------------
orig_img = nib.load(original_nifti)
orig_affine = orig_img.affine
orig_header = orig_img.header.copy()
nx, ny, nz = orig_img.shape  # (512, 512, z)

# --- Prepare constant -1000 background volume -------------------------------
sct_data = np.full((nx, ny, nz), -1000, dtype=np.float32)

# --- Gather and sort slice files --------------------------------------------
pattern = os.path.join(input_dir, "CBCT_slice_*.npy")
slice_files = sorted(
    glob.glob(pattern),
    key=lambda fn: int(os.path.basename(fn).split("_")[-1].split(".")[0])
)
if len(slice_files) != nz:
    raise ValueError(f"Expected {nz} slices, found {len(slice_files)}.")

# --- Reverse transform & insert ROI per slice -------------------------------
for idx, fn in enumerate(slice_files):
    # Load 256×256 prediction
    pred256 = np.load(fn)

    # 1) Upsample to 366×366
    patch366 = zoom(pred256, (zoom_factor, zoom_factor), order=ORDER)

    # 2) Remove padding → 340×238 ROI
    y_start, y_end = pad_top, pad_top + crop_h
    x_start, x_end = pad_left, pad_left + crop_w
    patchROI = patch366[y_start:y_end, x_start:x_end]
    assert patchROI.shape == (crop_h, crop_w), f"Got {patchROI.shape}"

    # 3) Insert into the -1000 background
    sct_data[y0:y1, x0:x1, idx] = patchROI

# --- Save as NIfTI ----------------------------------------------------------
sct_data = np.rot90(sct_data, k=1, axes=(0,1))
sct_img = nib.Nifti1Image(sct_data, orig_affine, orig_header)
nib.save(sct_img, output_filename)
print(f"✓ Saved sCT with -1000 background to '{output_filename}'")
