import os
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

input_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT'
output_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT_NPY'

# input_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT'
# output_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT_NPY'

os.makedirs(output_dir, exist_ok=True)

def load_nifti_image(file_path):
    image = nib.load(file_path)
    proxy = image.dataobj
    print("slope:", proxy.slope, "inter:", proxy.inter)
    return image.get_fdata()

def save_slices_as_npy(image_data, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    for i in range(image_data.shape[2]):  # Iterate over axial slices
        img_slice = image_data[:, :, i]
        output_file = os.path.join(output_path, f'slice_{i}.npy')
        _min, _max = image_data.min(), image_data.max()
        print("min:", _min, "max:", _max)
        np.save(output_file, img_slice)  # Save the slice as .npy file
    
    # Save metadata: Min/Max HU values (optional)
    hu_range_file = os.path.join(output_path, "hu_range.npy")
    np.save(hu_range_file, [_min, _max])

def process_nifti_file(file_name):
    file_path = os.path.join(input_dir, file_name)
    img_data = load_nifti_image(file_path)
    output_path = os.path.join(output_dir, file_name.split('.')[0])
    save_slices_as_npy(img_data, output_path)
    print(f"Processed {file_name}: Saved {img_data.shape[2]} slices to {output_path}")

if __name__ == "__main__":
    nifti_files = [f for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))]
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.map(process_nifti_file, nifti_files)
