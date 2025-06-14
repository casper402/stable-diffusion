#!/usr/bin/env python
import os
import numpy as np
import nibabel as nib

DEBUG = False
ROTATE = True
CLIP = False

def process_nifti_file(folder_path, nifty_file_name, output_dir):
    """
    Processes one 3D NIfTI file:
      - Loads the volume.
      - Computes statistics on each slice (based on original values): minimum, maximum, and gap.
      - Clips values to [-1000,1000], rotates each slice 90° clockwise, and saves each slice as a .npy file.
    Returns:
      base_name: the volume base name (derived from the file name, e.g. "volume-0")
      max_gap: largest gap (max-min) found in any slice (from original data)
      max_gap_idx: slice index with that gap
      global_min: lowest voxel value found (original data)
      global_min_idx: slice index where that lowest value was found
      global_max: highest voxel value found (original data)
      global_max_idx: slice index where that highest value was found
    """
    nifty_path = os.path.join(folder_path, nifty_file_name)
    print("Loading", nifty_path, "...")
    
    # Load the NIfTI file (assumed 3D)
    img = nib.load(nifty_path)
    data = img.get_fdata()         # original data for stats
    if CLIP:
        data_clipped = np.clip(data, -1000, 1000)  # clipped version for saving
    else:
        data_clipped = data
    
    # Create output directory if needed.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine a base name for the volume.
    if nifty_file_name.endswith('.nii.gz'):
        base_name = nifty_file_name[:-7]
    else:
        base_name = os.path.splitext(nifty_file_name)[0]
    
    # If the CBCT image name starts with "REC-", change it to "volume-"
    if base_name.startswith("REC-"):
        base_name = "volume-" + base_name[4:]
    
    # Initialize statistic trackers (computed on original data)
    max_gap = -np.inf
    max_gap_idx = None
    global_min = np.inf
    global_min_idx = None
    global_max = -np.inf
    global_max_idx = None

    num_slices = data.shape[2]
    for i in range(num_slices):
        # Statistics computed on the original slice (before clipping)
        initial_slice_data = data[:, :, i]
        smin = initial_slice_data.min()
        smax = initial_slice_data.max()
        gap = smax - smin

        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i
        if smin < global_min:
            global_min = smin
            global_min_idx = i
        if smax > global_max:
            global_max = smax
            global_max_idx = i

        # Rotate the clipped slice 90° clockwise (k=-1) and save it.
        output = data_clipped[:, :, i]
        if ROTATE:
            output = np.rot90(output, k=-1)
        slice_file = os.path.join(output_dir, f"{base_name}_slice_{i:03d}.npy")
        np.save(slice_file, output)
            
        if DEBUG:
            print(f"Saved rotated slice {i} as: {slice_file}")

    print(f"\nVolume {base_name}:")
    print(f"  Slice with biggest gap: {max_gap_idx} (gap = {max_gap})")
    print(f"  Slice with lowest value: {global_min_idx} (min = {global_min})")
    print(f"  Slice with highest value: {global_max_idx} (max = {global_max})")
    
    return base_name, max_gap, max_gap_idx, global_min, global_min_idx, global_max, global_max_idx

def process(volume_idx):
    # Change these paths as needed.
    folder_path   = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT"
    # For CBCT volumes, files are named like "REC-0.nii". The renaming in process_nifti_file
    # will ensure that the output base name is similar to "volume-0".
    nifty_file_name = f"volume-{volume_idx}.nii"
    output_dir = os.path.expanduser("/Users/Niklas/thesis/training_data/CT-unclipped")
    
    return process_nifti_file(folder_path, nifty_file_name, output_dir)

def procecss_batch():
    num_volumes = 131
    # Lists to store per-volume stats as tuples:
    # For gap: (max_gap, volume_idx, base_name, slice_index)
    gap_list = []
    # For lowest value: (global_min, volume_idx, base_name, slice_index)
    low_list = []
    # For highest value: (global_max, volume_idx, base_name, slice_index)
    high_list = []

    for i in range(num_volumes):
        try:
            result = process(i)
            base_name, max_gap, max_gap_idx, global_min, global_min_idx, global_max, global_max_idx = result
            # gap_list.append((max_gap, i, base_name, max_gap_idx))
            # low_list.append((global_min, i, base_name, global_min_idx))
            # high_list.append((global_max, i, base_name, global_max_idx))
        except Exception as e:
            print(f"Error processing volume {i}: {e}")

    # Sort and take the top 10.
    # top_gap = sorted(gap_list, key=lambda x: x[0], reverse=True)[:10]
    # top_low = sorted(low_list, key=lambda x: x[0])[:10]
    # top_high = sorted(high_list, key=lambda x: x[0], reverse=True)[:10]

    # print("\nTop 10 volumes with highest gap:")
    # for gap, vol_idx, base, slice_idx in top_gap:
    #     print(f"  Volume {vol_idx} ({base}): gap = {gap} at slice {slice_idx}")

    # print("\nTop 10 volumes with lowest value:")
    # for min_val, vol_idx, base, slice_idx in top_low:
    #     print(f"  Volume {vol_idx} ({base}): lowest value = {min_val} at slice {slice_idx}")

    # print("\nTop 10 volumes with highest value:")
    # for max_val, vol_idx, base, slice_idx in top_high:
    #     print(f"  Volume {vol_idx} ({base}): highest value = {max_val} at slice {slice_idx}")

if __name__ == "__main__":
    procecss_batch()
