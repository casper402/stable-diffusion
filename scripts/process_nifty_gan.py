import os
import numpy as np
import nibabel as nib

def process_nifti_file_normalized(folder_path, nifty_file_name, output_dir):
    nifty_path = os.path.join(folder_path, nifty_file_name)
    print(f"Loading {nifty_path}...")

    if not (nifty_file_name.endswith('.nii') or nifty_file_name.endswith('.nii.gz')):
        raise ValueError("The provided file is not a NIfTI file (.nii or .nii.gz).")

    img = nib.load(nifty_path)
    data = img.get_fdata()

    min_val = np.min(data)
    max_val = np.max(data)

    if max_val == min_val:
        normalized_data = np.zeros_like(data)
    else:
        normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if nifty_file_name.endswith('.nii.gz'):
        base_name = nifty_file_name[:-7]
    else:
        base_name = os.path.splitext(nifty_file_name)[0]

    if base_name.startswith("REC-"):
        base_name = "volume-" + base_name[4:]

    num_slices = normalized_data.shape[2]
    print(f"Processing {num_slices} slices...")

    # Define patch intervals for the *expected* (366, 238) slice shape
    # The first dimension (rows, corresponding to 366) will use the 4 intervals
    row_intervals = [(0, 120), (82, 202), (164, 284), (246, 366)]
    # The second dimension (columns, corresponding to 238) will use the 3 intervals
    col_intervals = [(0, 120), (59, 179), (118, 238)]

    for i in range(num_slices):
        slice_data = normalized_data[:, :, i]

        # Explicitly check if the slice has the expected (366, 238) shape
        if slice_data.shape != (366, 238):
            print(f"Warning: Slice {i:03d} has an unexpected shape {slice_data.shape}. Expected (366, 238). Patches might be incorrect.")

        patch_count = 0
        for row_start, row_end in row_intervals:
            for col_start, col_end in col_intervals:
                patch = slice_data[row_start:row_end, col_start:col_end]

                if patch.shape != (120, 120):
                    print(f"Warning: Patch from slice {i:03d}, Row-interval [{row_start},{row_end}], Col-interval [{col_start},{col_end}] has shape {patch.shape} instead of (120, 120). This indicates an underlying issue with the slice dimensions.")

                patch_file_path = os.path.join(output_dir, f"{base_name}_slice_{i:03d}_patch_{patch_count:02d}.npy")
                np.save(patch_file_path, patch)
                patch_count += 1
        print(f"Slice {i:03d}: Saved {patch_count} patches.")

    print(f"Finished processing volume '{base_name}'. Patches saved to '{output_dir}'")
    return base_name

def reconstruct_slice_from_patches(patches_dir, base_name, slice_index, target_shape=(366, 238)):
    """
    Reconstructs a single 2D slice from its 12 patches, handling overlaps by averaging.

    Args:
        patches_dir (str): The directory where the patches are saved.
        base_name (str): The base name of the volume (e.g., "volume-3").
        slice_index (int): The index of the slice to reconstruct (e.g., 0).
        target_shape (tuple): The expected shape of the reconstructed slice (height, width).

    Returns:
        numpy.ndarray: The reconstructed 2D slice.
    """
    reconstructed_slice = np.zeros(target_shape, dtype=np.float32)
    overlap_count = np.zeros(target_shape, dtype=np.int32)

    # Define patch intervals (must match how they were created)
    row_intervals = [(0, 120), (82, 202), (164, 284), (246, 366)]
    col_intervals = [(0, 120), (59, 179), (118, 238)]

    num_rows_patches = len(row_intervals) # 4
    num_cols_patches = len(col_intervals) # 3

    for patch_idx in range(num_rows_patches * num_cols_patches):
        # Calculate row and column index for the current patch
        row_grid_idx = patch_idx // num_cols_patches
        col_grid_idx = patch_idx % num_cols_patches

        row_start, row_end = row_intervals[row_grid_idx]
        col_start, col_end = col_intervals[col_grid_idx]

        patch_file_path = os.path.join(patches_dir, f"{base_name}_slice_{slice_index:03d}_patch_{patch_idx:02d}.npy")

        try:
            patch_data = np.load(patch_file_path)
            patch_data = np.squeeze(patch_data) # Ensure it's 2D (120, 120)

            # Add patch data to the accumulator
            reconstructed_slice[row_start:row_end, col_start:col_end] += patch_data

            # Increment overlap count for the region covered by this patch
            overlap_count[row_start:row_end, col_start:col_end] += 1

        except FileNotFoundError:
            print(f"Error: Patch file not found at {patch_file_path}. Skipping this patch.")
        except Exception as e:
            print(f"Error loading patch {patch_file_path}: {e}")

    # Handle areas where overlap_count might be zero (shouldn't happen with these intervals,
    # but good for robustness)
    # Use np.where to avoid division by zero
    final_slice = np.where(overlap_count > 0, reconstructed_slice / overlap_count, 0)
    
    # Ensure the reconstructed slice is within the normalized range (-1, 1)
    # Due to floating point arithmetic, slight deviations might occur.
    final_slice = np.clip(final_slice, -1.0, 1.0)

    print(f"Reconstructed slice {slice_index:03d} with shape {final_slice.shape}.")
    return final_slice

def post_process_single_volume():
    # Define paths and parameters directly within this function
    patches_dir = "/home/casper/Documents/Thesis/test/volume-3-gan-pred" # This should be where your PATCHES are stored
    volume_base_name = "volume-3" # This is the base name derived from your NIfTI file (e.g., REC-3.nii -> volume-3)
    output_dir = "/home/casper/Documents/Thesis/test/volume-3-stiched" # Where you want the RECONSTRUCTED slices saved
    n_slices = 364 # Total number of slices in your volume

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting reconstruction for {n_slices} slices of volume '{volume_base_name}'.")

    for slice_idx in range(n_slices):
        # Call the reconstruction function
        processed_slice = reconstruct_slice_from_patches(patches_dir, volume_base_name, slice_idx)
        
        # Define the output file path for the reconstructed slice
        output_file = os.path.join(output_dir, f"{volume_base_name}_slice_{slice_idx:03d}.npy")
        
        # Save the reconstructed slice
        np.save(output_file, processed_slice)
        print(f"Saved reconstructed slice {slice_idx:03d} to {output_file}")

    print(f"\nFinished reconstructing all {n_slices} slices for volume '{volume_base_name}'.")
    print(f"Reconstructed slices are saved in: {output_dir}")

def process_single_volume():
    folder_path = "/media/casper/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated/490"
    nifty_file_name = "REC-3.nii"
    output_dir = "/home/casper/Documents/Thesis/test_gan/volume-3"

    try:
        processed_base_name = process_nifti_file_normalized(folder_path, nifty_file_name, output_dir)
        print(f"\nSuccessfully processed: {processed_base_name}")
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print(f"Error: NIfTI file not found at {os.path.join(folder_path, nifty_file_name)}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    #process_single_volume()
    post_process_single_volume()