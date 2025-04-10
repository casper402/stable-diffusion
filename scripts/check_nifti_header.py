import nibabel as nib

input_dir = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TESTCTAlignedToCBCT'

def check_nifti_header(file_path):
    # Load the NIfTI file
    nifti_image = nib.load(file_path)
    
    # Get the header
    header = nifti_image.header
    
    # Print the entire header (optional)
    print("Full Header:")
    print(header)
    
    # Look for scaling information
    scl_slope = header.get('scl_slope', None)
    scl_inter = header.get('scl_inter', None)
    
    # Print scaling information if it exists
    if scl_slope is not None and scl_inter is not None:
        print("found some info")
        print(f"Scaling Slope (scl_slope): {scl_slope}")
        print(f"Scaling Intercept (scl_inter): {scl_inter}")
    else:
        print("No scaling information found (scl_slope or scl_inter).")
        print("scl_slope", scl_slope)
        print("sccl_inter", scl_inter)


    print("what about in the proxy dataobject??")
    proxy = nifti_image.dataobj
    print("slope:", proxy.slope)
    print("inter:", proxy.inter)

if __name__ == "__main__":
    file_path = input_dir + '/test-volume-0.nii'
    check_nifti_header(file_path)

