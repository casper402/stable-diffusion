import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

input_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT2D_NPY'
output_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT2D_NPY_Normalized'

# Fixed HU range for normalization
HU_MIN = -1000
HU_MAX = 1000

def normalize_hu_values(image_data):
    """Normalize the HU values to the range [-1, 1]."""
    return 2 * ((image_data - HU_MIN) / (HU_MAX - HU_MIN)) - 1

def process_npy_file(file_path, output_path):
    """Load, normalize, and save a single NPY file."""
    image_data = np.load(file_path)
    normalized_data = normalize_hu_values(image_data)
    np.save(output_path, normalized_data)

def process_directory(subdir):
    """Process all .npy files in a given subdirectory."""
    input_subdir = os.path.join(input_dir, subdir)
    output_subdir = os.path.join(output_dir, subdir)
    os.makedirs(output_subdir, exist_ok=True)
    
    npy_files = [f for f in os.listdir(input_subdir) if f.endswith('.npy') and f != "hu_range.npy"]
    
    for file_name in npy_files:
        file_path = os.path.join(input_subdir, file_name)
        output_path = os.path.join(output_subdir, file_name)
        process_npy_file(file_path, output_path)
    
    print(f"Processed {len(npy_files)} slices in {subdir}")

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_directory, subdirs)
    
    print("Normalization completed!")