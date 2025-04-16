#!/usr/bin/env python
import os
import numpy as np
import nibabel as nib
import concurrent.futures
from PIL import Image, ImageOps

DEBUG = True
SCALING_METHOD = "simple"  # Options: "simple", "zscore", "percentile"
PERCENTILE = 0.01

def scale_simple(data):
    """
    Naively scale the data from [0, 65535] to [-1000, 1000]:
         scaled = (data / 65535) * 2000 - 1000
    """
    return (data / 65535) * 2000 - 1000

def scale_zscore(data):
    """
    Computes the per-scan Z-score for the entire scan.
    Returns the z-score version of data (without clipping) where:
         z = (data - μ) / σ.
    """
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma == 0:
        return data - mu  # Avoid division by zero
    return (data - mu) / sigma

def scale_percentile(data, target_min=-1000, target_max=1000):
    """
    Robust Percentile-Based Linear Mapping.
    Computes the lower and upper thresholds at the specified percentiles.
    Clips the data to those thresholds and then linearly maps the values
    from [lo, hi] to [target_min, target_max].
    
    Returns the scaled data. (For debugging you can also return lo and hi.)
    """
    lo = np.percentile(data, lower_percentile)
    hi = np.percentile(data, upper_percentile)
    data_clipped = np.clip(data, lo, hi)
    scaled = (data_clipped - lo) / (hi - lo)
    scaled = scaled * (target_max - target_min) + target_min
    if DEBUG:
        print(f"Percentile thresholds: {lower_percentile}th = {lo:.2f}, {upper_percentile}th = {hi:.2f}")
    return scaled

def transform_slice(slice_array):
    """
    Transforms a 2D numpy array so that the output becomes a 256x256 image.
    The transformation is:
      - Convert to a PIL image (floating point mode).
      - Pad the image (e.g., pad (left, top, right, bottom) as needed).
      - Resize to 256x256 using bilinear interpolation.
    Returns a 256x256 numpy array.
    """
    img = Image.fromarray(slice_array).convert("F")
    padded_img = ImageOps.expand(img, border=(0, 64, 0, 64), fill=0)
    resized_img = padded_img.resize((256, 256), resample=Image.BILINEAR)
    out_array = np.array(resized_img)
    return out_array

def process_cbct_nifti_file(folder_path, nifty_file_name, output_dir, scaling_method="simple"):
    """
    Processes one 3D CBCT NIfTI file:
      - Loads the volume.
      - Scales the data using one of the following methods:
            * "simple": scales from [0, 65535] to [-1000, 1000]
            * "zscore": computes z = (x-μ)/σ, reports per-slice statistics on the raw z values,
                        clips them to [-1, 1], and then maps to [-1000,1000] (multiplying by 1000)
            * "percentile": clips the data at the 2nd and 98th percentiles and linearly maps to [-1000,1000]
      - For each slice, computes statistics on the scaled data.
      - Rotates each slice 90° clockwise, applies the transformation (padding+resize to 256x256),
        and saves it as a compressed file (.npz).
    
    For naming consistency, if the input file is named "REC-<num>.nii",
    the base name is changed to "volume-<num>".
    
    Returns:
      base_name, max_gap, max_gap_idx, global_min, global_min_idx, global_max, global_max_idx
    """
    nifty_path = os.path.join(folder_path, nifty_file_name)
    print("Loading", nifty_path, "...")
    
    # Load the NIfTI file (assumed to be 3D).
    img = nib.load(nifty_path)
    data = img.get_fdata()  # Original data in [0, 65535]
    
    if scaling_method == "simple":
        data_scaled = scale_simple(data)
    elif scaling_method == "zscore":
        data_z = scale_zscore(data)
    elif scaling_method == "percentile":
        data_scaled = scale_percentile(data)
    else:
        raise ValueError("Unknown scaling method: " + scaling_method)
    
    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine the base name.
    if nifty_file_name.endswith('.nii.gz'):
        base_name = nifty_file_name[:-7]
    else:
        base_name = os.path.splitext(nifty_file_name)[0]
    if base_name.startswith("REC-"):
        base_name = "volume-" + base_name[4:]
    
    # Initialize statistic trackers.
    max_gap = -np.inf
    max_gap_idx = None
    global_min = np.inf
    global_min_idx = None
    global_max = -np.inf
    global_max_idx = None

    num_slices = data.shape[2]
    for i in range(num_slices):
        if scaling_method in ["simple", "percentile"]:
            slice_scaled = data_scaled[:, :, i]
            smin = slice_scaled.min()
            smax = slice_scaled.max()
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
        elif scaling_method == "zscore":
            slice_z = data_z[:, :, i]
            slice_min = slice_z.min()
            slice_max = slice_z.max()
            slice_gap = slice_max - slice_min
            if slice_gap > max_gap:
                max_gap = slice_gap
                max_gap_idx = i
            if slice_min < global_min:
                global_min = slice_min
                global_min_idx = i
            if slice_max > global_max:
                global_max = slice_max
                global_max_idx = i
            slice_scaled = np.clip(slice_z, -1, 1) * 1000
        else:
            slice_scaled = None
        
        # Rotate 90° clockwise.
        rotated_slice = np.rot90(slice_scaled, k=-1)
        # Transform to a 256x256 image.
        transformed_slice = transform_slice(rotated_slice)
        # Save as a compressed .npz file.
        # Use a .npz extension and store the array with key "slice".
        slice_file = os.path.join(output_dir, f"{base_name}_slice_{i:03d}.npz")
        np.savez_compressed(slice_file, slice=transformed_slice)
        if DEBUG:
            print(f"Saved transformed slice {i} as: {slice_file}")
    
    if DEBUG:
        print(f"\nVolume {base_name}:")
        print(f"  Slice with biggest gap: {max_gap_idx} (gap = {max_gap})")
        print(f"  Slice with lowest value: {global_min_idx} (min = {global_min})")
        print(f"  Slice with highest value: {global_max_idx} (max = {global_max})")
    
    return base_name, max_gap, max_gap_idx, global_min, global_min_idx, global_max, global_max_idx

def process(volume_idx, scaling_method=SCALING_METHOD):
    folder_path = os.path.expanduser("/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated/256")
    nifty_file_name = f"REC-{volume_idx}.nii"  # CBCT files like "REC-0.nii", etc.
    post = ""
    if scaling_method == "percentile":
        post = "/" + str(PERCENTILE)
    output_dir = os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/" + scaling_method + post)
    return process_cbct_nifti_file(folder_path, nifty_file_name, output_dir, scaling_method=scaling_method)

def process_batch(scaling_method=SCALING_METHOD):
    num_volumes = 131
    gap_list = []
    low_list = []
    high_list = []
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_volume = {executor.submit(process, i, scaling_method=scaling_method): i for i in range(num_volumes)}
        for future in concurrent.futures.as_completed(future_to_volume):
            vol_idx = future_to_volume[future]
            try:
                result = future.result()
                if result is not None:
                    base_name, max_gap, max_gap_idx, global_min, global_min_idx, global_max, global_max_idx = result
                    results[vol_idx] = (base_name, max_gap, max_gap_idx, global_min, global_min_idx, global_max, global_max_idx)
                    gap_list.append((max_gap, vol_idx, base_name, max_gap_idx))
                    low_list.append((global_min, vol_idx, base_name, global_min_idx))
                    high_list.append((global_max, vol_idx, base_name, global_max_idx))
                    print("Finished volume", vol_idx)
            except Exception as e:
                print(f"Error processing volume {vol_idx}: {e}")
    
    for vol_idx in sorted(results.keys()):
        base_name, max_gap, max_gap_idx, global_min, global_min_idx, global_max, global_max_idx = results[vol_idx]
        print(f"\nVolume {vol_idx} ({base_name}):")
        print(f"  Slice with biggest gap: {max_gap_idx} (gap = {max_gap})")
        print(f"  Slice with lowest value: {global_min_idx} (min = {global_min})")
        print(f"  Slice with highest value: {global_max_idx} (max = {global_max})")
    
    top_gap = sorted(gap_list, key=lambda x: x[0], reverse=True)[:10]
    top_low = sorted(low_list, key=lambda x: x[0])[:10]
    top_high = sorted(high_list, key=lambda x: x[0], reverse=True)[:10]
    
    print("\nTop 10 volumes with highest gap:")
    for gap, vol_idx, base, slice_idx in top_gap:
        print(f"  Volume {vol_idx} ({base}): gap = {gap} at slice {slice_idx}")
    
    print("\nTop 10 volumes with lowest value:")
    for min_val, vol_idx, base, slice_idx in top_low:
        print(f"  Volume {vol_idx} ({base}): lowest value = {min_val} at slice {slice_idx}")
    
    print("\nTop 10 volumes with highest value:")
    for max_val, vol_idx, base, slice_idx in top_high:
        print(f"  Volume {vol_idx} ({base}): highest value = {max_val} at slice {slice_idx}")

if __name__ == "__main__":
    # Uncomment one of the lines below to run with the desired scaling method.

    # For naive scaling:
    # process_batch(scaling_method="simple")

    # For Z-score scaling with clipping to [-1,1] then mapping to [-1000,1000]:
    # process_batch(scaling_method="zscore")

    # For robust percentile based scaling:
    # process_batch(scaling_method="percentile")

    # To process single batch:
    process(0)
