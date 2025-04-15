#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from evaluation import compute_mae, compute_rmse

def plot_multi_side_by_side(test_dirs, gt_dir, volume_idx, slice_num):
    """
    Plots the ground truth image and an arbitrary number of test images in a grid
    with up to 4 plots per row. Also computes MAE and RMSE for each test image relative 
    to the ground truth and adds these values to each test image's subplot title.
    
    As you move the mouse over the images, an annotation updates to show the pixel value
    at the cursor for the ground truth and, for each test image, its pixel value along with 
    the difference (Δ) from the ground truth.
    
    Parameters:
      test_dirs (list of str): Directories containing the test images.
      gt_dir (str): Directory containing the ground truth images.
      volume_idx (int): The volume index.
      slice_num (int): The slice index (0-indexed).
    
    Assumes the naming scheme:
        volume-<volume_idx>_slice_<slice_num:03d>.npy
    """
    # Construct the common filename.
    base_name = f"volume-{volume_idx}"
    filename = f"{base_name}_slice_{slice_num:03d}.npy"
    
    # Load ground truth image.
    gt_file = os.path.join(gt_dir, filename)
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    gt_image = np.load(gt_file)
    
    # Load test images from all test directories.
    test_images = []
    test_names = []
    for test_dir in test_dirs:
        test_file = os.path.join(test_dir, filename)
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found in {test_dir}: {test_file}")
        test_images.append(np.load(test_file))
        test_names.append(os.path.basename(os.path.normpath(test_dir)))
    
    num_tests = len(test_images)
    total_plots = 1 + num_tests  # one for GT plus one for each test image

    # Set maximum 4 columns per row.
    ncols = 4 if total_plots >= 4 else total_plots
    nrows = (total_plots + ncols - 1) // ncols  # ceiling division for rows

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 6 * nrows))
    # Flatten axes array for easy indexing.
    axes = axes.flatten()
    
    # Hide any unused subplots.
    for j in range(total_plots, len(axes)):
        axes[j].axis("off")

    # Plot the ground truth image in the first subplot.
    axes[0].imshow(gt_image, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    
    # For each test image, compute error metrics relative to the ground truth,
    # then plot and add the metrics to the subplot title.
    for i, test_img in enumerate(test_images, start=1):
        mae = compute_mae(test_img, gt_image)
        rmse = compute_rmse(test_img, gt_image)
        axes[i].imshow(test_img, cmap="gray")
        axes[i].set_title(f"Test: {test_names[i-1]}\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
        axes[i].axis("off")
    
    # Create an annotation text at the bottom of the figure.
    info_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=10)
    
    # Callback to update the annotation with pixel intensity differences.
    def on_move(event):
        if event.inaxes is None:
            info_text.set_text("")
            fig.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            info_text.set_text("")
            fig.canvas.draw_idle()
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        h, w = gt_image.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            info_text.set_text("")
            fig.canvas.draw_idle()
            return

        gt_val = gt_image[y, x]
        text_parts = [f"({x}, {y}) | GT: {gt_val:.2f}"]
        for idx, test_img in enumerate(test_images):
            test_val = test_img[y, x]
            diff = test_val - gt_val
            text_parts.append(f"{test_names[idx]}: {test_val:.2f} (Δ={diff:+.2f})")
        info_text.set_text(" | ".join(text_parts))
        fig.canvas.draw_idle()
    
    # Connect the motion event.
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    
    fig.suptitle(f"Volume {volume_idx} - Slice {slice_num}", fontsize=16)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show()

def main():
    test_dirs = [
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/simple"),
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/percentile/0.01"),
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/percentile/0.02"),
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/percentile/0.03"),
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/percentile/0.04"),
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/percentile/0.05"),
        os.path.expanduser("/Users/Niklas/thesis/training_data/CBCT/percentile/0.1"),
    ]
    gt_dir = os.path.expanduser("/Users/Niklas/thesis/training_data/CT")
    
    volume_idx = 0
    slice_num = 10
    plot_multi_side_by_side(test_dirs, gt_dir, volume_idx, slice_num)

if __name__ == "__main__":
    main()
