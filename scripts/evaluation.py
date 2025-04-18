#!/usr/bin/env python
import os
import glob
import numpy as np
import concurrent.futures

DEBUG = False

def compute_mae(test_image, gt_image):
    """
    Compute the Mean Absolute Error (MAE) between two images.
    """
    assert test_image.shape == gt_image.shape, "Images must have the same shape!"
    return np.mean(np.abs(test_image - gt_image))

def compute_rmse(test_image, gt_image):
    """
    Compute the Root Mean Square Error (RMSE) between two images.
    """
    assert test_image.shape == gt_image.shape, "Images must have the same shape!"
    return np.sqrt(np.mean((test_image - gt_image) ** 2))

def compare_single(filename, test_folder, gt_folder):
    """
    Load a test image and its corresponding ground truth image (both stored as .npy files)
    using the given filename, and compute the MAE and RMSE.
    """
    test_path = os.path.join(test_folder, filename)
    gt_path   = os.path.join(gt_folder, filename)
    
    test_image = np.load(test_path)
    gt_image   = np.load(gt_path)
    
    mae_value = compute_mae(test_image, gt_image)
    rmse_value = compute_rmse(test_image, gt_image)
    return mae_value, rmse_value

def compare_batch(volume_idx, test_folder, gt_folder):
    """
    For the given volume index, find all files matching the pattern:
        volume-<volume_idx>_slice_*.npy
    in the test folder (and assume the ground truth folder has the same filenames).
    Compute the MAE and RMSE for each slice and return the averages for this volume.
    """
    pattern = os.path.join(test_folder, f"volume-{volume_idx}_slice_*.npy")
    file_list = sorted(glob.glob(pattern))
    
    if not file_list:
        print(f"No files found for volume {volume_idx} in {test_folder}")
        return None

    mae_list = []
    rmse_list = []
    for file_path in file_list:
        filename = os.path.basename(file_path)
        mae, rmse = compare_single(filename, test_folder, gt_folder)
        mae_list.append(mae)
        rmse_list.append(rmse)
    
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)

    # Optionally also print per volume here (if you want early feedback)
    if DEBUG:
        print(f"Volume {volume_idx}: {len(file_list)} slices;  Average MAE = {avg_mae:.2f}, Average RMSE = {avg_rmse:.2f}")

    return avg_mae, avg_rmse

def compare_all_volumes(volumes, test_folder, gt_folder):
    """
    Evaluates each volume (from index 0 to num_volumes-1) concurrently.
    After all threads are finished, prints the individual results in the correct order,
    then prints overall averages across all volumes.
    """
    all_results = {}  # Dictionary to store {volume_idx: (avg_mae, avg_rmse)}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all volume tasks.
        future_to_vol = {executor.submit(compare_batch, vol_idx, test_folder, gt_folder): vol_idx
                         for vol_idx in volumes}
        
        # Wait for all futures to complete and store the results.
        for future in concurrent.futures.as_completed(future_to_vol):
            vol_idx = future_to_vol[future]
            try:
                result = future.result()
                if result is not None:
                    all_results[vol_idx] = result
                    print("Finished volume ", vol_idx)
            except Exception as exc:
                print(f"Volume {vol_idx} generated an exception: {exc}")
    
    # Now print the individual metrics in order by volume index.
    for vol_idx in sorted(all_results.keys()):
        avg_mae, avg_rmse = all_results[vol_idx]
        print(f"Volume {vol_idx}: Average MAE = {avg_mae:.2f}, Average RMSE = {avg_rmse:.2f}")
    
    # Compute overall averages.
    if all_results:
        overall_avg_mae = np.mean([res[0] for res in all_results.values()])
        overall_avg_rmse = np.mean([res[1] for res in all_results.values()])
        print("\nOverall average across volumes:")
        print(f"  MAE  = {overall_avg_mae:.2f}")
        print(f"  RMSE = {overall_avg_rmse:.2f}")
    else:
        print("No volume results to summarize.")

if __name__ == "__main__":
    # Set your test and ground truth directories (adjust these paths as needed)
    test_folder = os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/scaledV2")
    gt_folder   = os.path.expanduser("/Users/Niklas/thesis/training_data/CT")
    
    # Set the number of volumes to process.
    # num_volumes = 131

    # Evaluate these volumes:
    volumes = [68, 27, 52, 104, 130, 16, 24, 75, 124, 26, 64, 90, 50, 86, 122, 106, 65, 62, 128, 69, 15, 117, 96, 3, 76, 109, 18, 120, 73, 79, 83, 14, 58, 17, 112, 13, 110, 125, 1, 126, 93, 51, 107, 91, 85, 82, 67, 102, 94, 56, 84, 53, 100, 11, 48, 101, 57, 55, 80, 39, 5, 49, 78, 129, 123, 7, 10, 88, 121, 95, 127, 92, 105, 116, 6, 19, 115, 97, 2, 118, 66, 54, 25, 63, 108, 22, 113, 8, 111, 114, 9, 74, 21, 77, 20, 103, 70, 87, 119, 4]
    
    compare_all_volumes(volumes, test_folder, gt_folder)
