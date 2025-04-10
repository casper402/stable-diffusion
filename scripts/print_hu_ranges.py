import os
import numpy as np

input_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT2D_NPY'

def print_hu_ranges():
    """Iterate over all hu_range.npy files and print the min/max HU values."""
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for subdir in subdirs:
        hu_range_path = os.path.join(input_dir, subdir, "hu_range.npy")
        if os.path.exists(hu_range_path):
            hu_min, hu_max = np.load(hu_range_path)
            print(f"{subdir}: Min HU = {hu_min}, Max HU = {hu_max}")
        else:
            print(f"{subdir}: hu_range.npy not found")

if __name__ == "__main__":
    print_hu_ranges()