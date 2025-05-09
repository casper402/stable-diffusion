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
input_dir = os.path.expanduser("/Users/Niklas/thesis/predictions/prediction-clinic-1stepsize")
original_nifti = os.path.expanduser("/Users/Niklas/thesis/training_data/Casper-klinik/CBCT.nii.gz")
output_filename = os.path.expanduser(f"/Users/Niklas/thesis/training_data/Casper-klinik/sCT-1000-steps-{ORDER}.nii.gz")

# --- Load original geometry -------------------------------------------------
orig_img = nib.load(original_nifti)
orig_affine = orig_img.affine
orig_header = orig_img.header.copy()

# --- Locate and sort slice files --------------------------------------------
pattern = os.path.join(input_dir, 'CBCT_slice_*.npy')
slice_files = glob.glob(pattern)
if not slice_files:
    raise FileNotFoundError(f"No files found with pattern: {pattern}")

def get_index(path):
    """Extract the slice index from 'CBCT_slice_###.npy'"""
    name = os.path.basename(path)
    num_str = name.replace('CBCT_slice_', '').replace('.npy', '')
    return int(num_str)

slice_files_sorted = sorted(slice_files, key=get_index)
print(f"Found {len(slice_files_sorted)} slices. Stacking...")

# --- Load and stack ---------------------------------------------------------
slices = [np.load(f) for f in slice_files_sorted]
volume = np.stack(slices, axis=2)  # shape: (256, 256, z)
volume = np.rot90(volume, k=1, axes=(0,1))
print(f"Loaded volume shape: {volume.shape}")

# --- Resize back to original in-plane resolution ---------------------------
# From (256,256,z) back to (512,512,z) => zoom factors (2, 2, 1)
zoom_factors = (2, 2, 1)
resized_volume = zoom(volume, zoom_factors, order=ORDER)  # nearest-neighbor
print(f"Resized volume shape: {resized_volume.shape}")

# --- Save as NIfTI with original geometry ----------------------------------
# The header's pixdim fields remain consistent with the original 512×512 voxels
resized_img = nib.Nifti1Image(resized_volume, orig_affine, orig_header)
nib.save(resized_img, output_filename)
print(f"Saved resized sCT volume to '{output_filename}'")
