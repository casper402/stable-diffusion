import os
import nibabel as nib
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Casper
# input_dir = '/media/casper/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT'
# output_dir = '/media/casper/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT2D'

# Niklas
input_dir = os.path.expanduser('~/Downloads/cbct-liver-and-liver-tumor-segmentation-train-data/TRAINCTAlignedToCBCT')
output_dir = os.path.expanduser('~/thesis/images/TRAINCTAlignedToCBCT2D')

os.makedirs(output_dir, exist_ok=True)

def load_nifti_image(file_path):
    image = nib.load(file_path)
    return image.get_fdata()

def save_slices_as_png(image_data, output_path):
    plt.rcParams['image.cmap'] = 'gray'  
    plt.rcParams['image.interpolation'] = 'nearest'

    os.makedirs(output_path, exist_ok=True)
    
    for i in range(image_data.shape[2]):  # Iterate over the third dimension
        img_slice = image_data[:, :, i]
        output_file = os.path.join(output_path, f'slice_{i}.png')
        plt.imsave(output_file, img_slice.T)  # Transpose to match orientation

def process_nifti_file(file_name)
    file_path = os.path.join(input_dir, file_name)
    img_data = load_nifti_image(file_path)
    output_path = os.path.join(output_dir, file_name.split('.')[0])
    save_slices_as_png(img_data, output_path)
    print(f"Processed {file_name}: Saved {img_data.shape[2]} slices to {output_path}")

if __name__ == "__main__":
    nifti_files = [f for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))]
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_nifti_file, nifti_files)