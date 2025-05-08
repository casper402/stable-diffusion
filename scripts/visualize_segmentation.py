import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision

from utils.dataset import get_dataloaders, CTDatasetNPY, PairedCTCBCTDatasetNPY, PairedCTCBCTSegmentationDatasetNPY, SegmentationMaskDatasetNPY
from models.diffusion import Diffusion
from quick_loop.vae import load_vae, train_vae
from quick_loop.unet import load_unet, train_unet, train_joint
from quick_loop.unetConditional import load_cond_unet, train_cond_unet
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from quick_loop.unetControlPACA import load_unet_control_paca, train_dr_control_paca, test_dr_control_paca

### CONFIG ###
train_size = 10
val_size = None
test_size = 10
batch_size = 8
num_workers = 8
epochs = 2000
early_stopping = 70
patience = 30
epochs_between_prediction = 5
base_channels = 256
dropout_rate = 0.1
learning_rate = 5e-5
warmup_lr = 0
warmup_epochs = 0

# Augmentation
augmentation = {
    'degrees': (-1, 1),
    'translate': (0.1, 0.1),
    'scale': (0.9, 1.1),
    'shear': None,
}
augmentation = None

# Vae Loss params
perceptual_weight=0.05
ssim_weight=1
mse_weight=1.0
kl_weight=0.000001
l1_weight=0

# Load pretrained model paths
load_dir = "../pretrained_models"
load_vae_path = os.path.join(load_dir, "vae_new_loss_term.pth")
load_unet_path = os.path.join(load_dir, "unet.pth")
load_dr_module_path = os.path.join(load_dir, "dr_module-1819.pth")

# Save prediction / model directories
save_dir = "controlnet_new_no_augmentation"
os.makedirs(save_dir, exist_ok=True)
vae_predict_dir = os.path.join(save_dir, "vae_predictions")
unet_predict_dir = os.path.join(save_dir, "unet_predictions")
conditional_predict_dir = os.path.join(save_dir, "conditional_predictions")
vae_save_path = os.path.join(save_dir, "vae.pth")
unet_save_path = os.path.join(save_dir, "unet.pth")
controlnet_save_path = os.path.join(save_dir, "controlnet.pth")
paca_layers_save_path = os.path.join(save_dir, "paca_layers.pth")
degradation_removal_save_path = os.path.join(save_dir, "dr_module.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
manifest_path = "../training_data/manifest-filtered.csv" # without CBCT

train_loader, _, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=SegmentationMaskDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)

try:
    batch_data = next(iter(train_loader))
    segmentation_map_batch = batch_data[0]
    liver_mask_batch = batch_data[1]
    tumor_mask_batch = batch_data[2]
except StopIteration:
    print("Error: The test_loader is empty. Cannot retrieve a batch.")
    exit()
except IndexError:
    print("Error: Batch does not contain enough elements. Expected at least 5.")
    exit()
except Exception as e:
    print(f"An error occurred while retrieving batch data: {e}")
    exit()

if segmentation_map_batch.shape[0] == 0 or liver_mask_batch.shape[0] == 0 or tumor_mask_batch.shape[0] == 0:
    print("Batch is empty or one of the tensors is empty.")
    exit()

num_items_to_visualize = min(batch_size, len(segmentation_map_batch))

for i in range(num_items_to_visualize):
    segmentation_map_tensor = segmentation_map_batch[i]
    liver_mask_tensor = liver_mask_batch[i]
    tumor_mask_tensor = tumor_mask_batch[i]

    segmentation_map_np = segmentation_map_tensor.cpu().numpy()
    liver_mask_np = liver_mask_tensor.cpu().numpy()
    tumor_mask_np = tumor_mask_tensor.cpu().numpy()

    segmentation_map_np = (segmentation_map_np + 1) / 2.0

    if segmentation_map_np.ndim == 3 and segmentation_map_np.shape[0] == 1:
        segmentation_map_np = segmentation_map_np.squeeze(0)
    if liver_mask_np.ndim == 3 and liver_mask_np.shape[0] == 1:
        liver_mask_np = liver_mask_np.squeeze(0)
    if tumor_mask_np.ndim == 3 and tumor_mask_np.shape[0] == 1:
        tumor_mask_np = tumor_mask_np.squeeze(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(segmentation_map_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Segmentation Map')
    axes[0].axis('off')

    axes[1].imshow(liver_mask_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Liver Mask')
    axes[1].axis('off')

    axes[2].imshow(tumor_mask_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Tumor Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

print(f"Visualized {num_items_to_visualize} segmentation maps and masks from the batch.")