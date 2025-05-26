import matplotlib.pyplot as plt
import numpy as np

# Example: Create a sample 2D array (e.g., a grayscale image) and save it
# data_2d = np.random.rand(64, 64) * 255 # Simulate pixel values 0-255
# data_2d = data_2d.astype(np.uint8)
# np.save('data_2d.npy', data_2d)

# Load the .npy file
data_1 = np.load('base3.npy') # Assuming you have a data_2d.npy file
data_1 = np.squeeze(data_1)

data_2 = np.load('base_cbct3.npy')
data_2 = np.squeeze(data_2)

fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # figsize adjusts the total size of the figure
im1 = axes[0].imshow(data_1, cmap='gray', vmax=1)
im2 = axes[1].imshow(data_2, cmap='gray', vmax=1)
plt.show()

#plt.imshow(data_2, cmap='gray') # Use 'gray' colormap for grayscale images
# plt.title('2D Data Visualization (Image)')
# plt.colorbar(label='Value')