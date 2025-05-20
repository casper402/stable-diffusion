#!/usr/bin/env python3
"""
Postprocess CBCT-to-sCT output back to original geometry:

1. Load .npy slice files named CBCT_slice_###.npy
2. Sort them by slice index
3. Stack into a 3D volume of shape (256, 256, z)
4. Resize X/Y back to (512, 512, z) using nearest-neighbor interpolation
5. Save the result as a NIfTI, preserving the original CBCT affine and header

Edit the input paths and filenames below as needed.
"""

import os
import glob
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

# --- User settings -----------------------------------------------------------
ORDER = 1
volume_idx = 26

input_dir = os.path.expanduser(f"/Users/Niklas/thesis/predictions/predictions-v3-stepsize1/volume-{volume_idx}")
original_nifti   = f"/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT/volume-{volume_idx}.nii"
output_filename = os.path.expanduser(f"/Users/Niklas/thesis/training_data/Casper-klinik/sCT-volume_{volume_idx}-1000-steps-{ORDER}.nii")

# --- Load original geometry -------------------------------------------------
orig_img = nib.load(original_nifti)
orig_affine = orig_img.affine
orig_header = orig_img.header.copy()

# --- Locate and sort slice files --------------------------------------------
pattern = os.path.join(input_dir, '*_slice_*.npy')
slice_files = glob.glob(pattern)
if not slice_files:
    raise FileNotFoundError(f"No files found with pattern: {pattern}")

def get_index(path):
    name = os.path.basename(path)
    num_str = name.replace(f"volume-{volume_idx}_slice_", '').replace('.npy', '')
    return int(num_str)

slice_files_sorted = sorted(slice_files, key=get_index)
print(f"Found {len(slice_files_sorted)} slices. Stacking...")

# --- Load and stack ---------------------------------------------------------
slices = [np.load(f) for f in slice_files_sorted]


# ─── reverse the PyTorch transforms ────────────────────────────────────────────
ORIG_H, ORIG_W = 238, 366
RES_H, RES_W = 256, 256
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
_pad_h = ORIG_H + PAD_T + PAD_B   # 238 + 64 + 64 = 366
_pad_w = ORIG_W + PAD_L + PAD_R   # 366 + 0 + 0   = 366

# load model outputs
slices = [np.load(f) for f in slice_files_sorted]   # each is (256,256)

# compute scale factors
scale_y = _pad_h / RES_H     # 366/256 ≈ 1.4297
scale_x = _pad_w / RES_W     # 366/256 ≈ 1.4297

# un‐resize and un‐pad
rev_slices = []
for s in slices:
    # 1) up‐sample back to the padded size
    s_up = zoom(s, (scale_y, scale_x), order=0)            # now (366,366)
    # 2) crop off the top/bottom and left/right
    s_crop = s_up[PAD_T:PAD_T + ORIG_H, PAD_L:PAD_L + ORIG_W]  
    #        [ 64 : 64+238 ,   0 : 0+366 ]  → (238,366)
    rev_slices.append(s_crop.astype(s.dtype))


volume = np.stack(rev_slices, axis=2)  # shape: (238, 366, z)
volume = np.rot90(volume, k=1, axes=(0,1))
print(f"Loaded volume shape: {volume.shape}")

# --- Save as NIfTI with original geometry ----------------------------------
# The header's pixdim fields remain consistent with the original 512×512 voxels
resized_img = nib.Nifti1Image(volume, orig_affine, orig_header)
nib.save(resized_img, output_filename)
print(f"Saved resized sCT volume to '{output_filename}'")
