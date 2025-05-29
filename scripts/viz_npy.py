import matplotlib.pyplot as plt
import numpy as np
import os

# Example: Create a sample 2D array (e.g., a grayscale image) and save it
# data_2d = np.random.rand(64, 64) * 255 # Simulate pixel values 0-255
# data_2d = data_2d.astype(np.uint8)
# np.save('data_2d.npy', data_2d)

# Load the .npy file
def visualize_npy_files():
    load_path = "/home/casper/Documents/Thesis/test/volume-3-cbct"

    path_1 = f"{load_path}/volume-3_slice_000.npy"
    data_1 = np.load(path_1) # Assuming you have a data_2d.npy file
    data_1 = np.squeeze(data_1)

    load_path = "/home/casper/Documents/Thesis/test/volume-3-stiched"

    path_2 = f"{load_path}/volume-3_slice_000.npy"
    data_2 = np.load(path_2)
    data_2 = np.squeeze(data_2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # figsize adjusts the total size of the figure
    im1 = axes[0].imshow(data_1, cmap='gray', vmax=1)
    im2 = axes[1].imshow(data_2, cmap='gray', vmax=1)
    plt.show()

def visualize_all_patches():
    volume_name = "volume-3-gan-pred" # Name of the volume to visualize
    load_path = f"/home/casper/Documents/Thesis/test/{volume_name}" 
    slice_index = 0 

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    # Loop through all 12 patches and display them
    for i in range(12):
        patch_path = os.path.join(load_path, f"volume-3_slice_{slice_index:03d}_patch_{i:02d}.npy")
        
        try:
            data_patch = np.load(patch_path)
            data_patch = np.squeeze(data_patch) # Remove single-dimensional entries from the shape
            
            # Display the patch
            im = axes[i].imshow(data_patch, cmap='gray', vmin=-1, vmax=1) # vmin/vmax for normalized data
            axes[i].set_title(f"Patch {i:02d}") # Set a title for each subplot
            axes[i].axis('off') # Turn off axis labels for cleaner visualization
            
        except FileNotFoundError:
            print(f"Error: Patch file not found at {patch_path}. Skipping.")
            axes[i].set_title(f"Patch {i:02d} (Missing)")
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading or displaying {patch_path}: {e}")
            axes[i].set_title(f"Patch {i:02d} (Error)")
            axes[i].axis('off')

    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show()
#plt.imshow(data_2, cmap='gray') # Use 'gray' colormap for grayscale images
# plt.title('2D Data Visualization (Image)')
# plt.colorbar(label='Value')

if __name__ == "__main__":
    visualize_all_patches()
    #visualize_npy_files()