import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision

from utils.dataset import get_dataloaders, CTDatasetNPY, PairedCTCBCTDatasetNPY
from models.diffusion import Diffusion
from quick_loop.vae import load_vae, train_vae
from quick_loop.unet import load_unet, train_unet
from quick_loop.unetConditional import load_cond_unet, train_cond_unet
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from quick_loop.unetControlPACA import load_unet_control_paca, train_dr_control_paca, test_dr_control_paca

### CONFIG ###
train_size = None
val_size = None
test_size = 10
batch_size = 32
accumulation_steps = 1 # Effectively increases batch size to batch_size * accumulation_steps
num_workers = 8
epochs = 2000
early_stopping = 50
patience = 10
epochs_between_prediction = 5
base_channels = 256
dropout_rate = 0.1
augmentation = True # NOTE: Set augmentation parameters manually in dataset.py

# Load pretrained model paths
load_dir = "../pretrained_models"
load_vae_path = os.path.join(load_dir, "vae.pth")

# Save prediction / model directories
save_dir = "unet_base_channels_256_v2"
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
manifest_path = "../manifest-full.csv"
manifest_path = "../data_quick_loop/manifest.csv" # Local config

# vae = load_vae(load_vae_path, trainable=False)

# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=CTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# train_vae(vae=vae, train_loader=train_loader, val_loader=val_loader, epochs=epochs, early_stopping=early_stopping, patience=patience, save_path=vae_save_path, predict_dir=vae_predict_dir)
# unet = load_unet(trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)
# train_unet(unet=unet, 
#            vae=vae, 
#            train_loader=train_loader, 
#            val_loader=val_loader,
#            test_loader=test_loader, 
#            epochs=epochs, 
#            early_stopping=early_stopping, 
#            patience=patience, 
#            save_path=unet_save_path, 
#            predict_dir=unet_predict_dir,
#            epochs_between_prediction=epochs_between_prediction,
# )

# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)

# unet = load_cond_unet(trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)
# train_cond_unet(
#     unet=unet, 
#     vae=vae, 
#     train_loader=train_loader, 
#     val_loader=val_loader,
#     test_loader=test_loader, 
#     epochs=epochs, 
#     early_stopping=early_stopping, 
#     patience=patience, 
#     save_path=unet_save_path, 
#     predict_dir=unet_predict_dir,
#     epochs_between_prediction=epochs_between_prediction,
# )

# vae = load_vae(save_path=vae_save_path, trainable=False)
# unet = load_unet_control_paca(unet_save_path=unet_save_path, paca_trainable=True)
# controlnet = load_controlnet(save_path=unet_save_path, trainable=True)
# dr_module = load_degradation_removal(trainable=True)
# unet = load_unet_control_paca(unet_save_path=unet_save_path, paca_trainable=True)
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size)
# train_dr_control_paca(
#     vae=vae, 
#     unet=unet, 
#     controlnet=controlnet, 
#     dr_module=dr_module, 
#     train_loader=train_loader, 
#     val_loader=val_loader, 
#     epochs=epochs, 
#     save_dir=save_dir, 
#     predict_dir=conditional_predict_dir, 
#     early_stopping=early_stopping, 
#     patience=patience, 
#     epochs_between_prediction=50, 
#     accumulation_steps=accumulation_steps)

# _, _, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size)
# vae = load_vae(vae_save_path)
# unet = load_unet_control_paca(unet_save_path=unet_save_path, paca_save_path=paca_layers_save_path)
# control_net = load_controlnet(save_path=controlnet_save_path)
# dr_module = load_degradation_removal(save_path=degradation_removal_save_path)
# test_dr_control_paca(
#     vae=vae, 
#     unet=unet, 
#     controlnet=control_net, 
#     dr_module=dr_module, 
#     test_loader=test_loader, 
#     guidance_scales=[1.0],
#     num_images_to_save=100
# )

# import numpy as np
# import matplotlib.pyplot as plt
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# for (ct, cbct) in train_loader:
#     ct_image = ct[0].squeeze().numpy()  # Assuming ct and cbct are PyTorch tensors
#     cbct_image = cbct[0].squeeze().numpy() # Assuming cbct had a typo: squueze -> squeeze

#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 1 row and 2 columns

#     # Display CT image in the first subplot
#     im1 = axes[0].imshow(ct_image, cmap='gray', vmin=-1, vmax=1)
#     axes[0].set_title('CT Image')
#     axes[0].axis('off')

#     # Display CBCT image in the second subplot
#     im2 = axes[1].imshow(cbct_image, cmap='gray', vmin=-1, vmax=1)
#     axes[1].set_title('CBCT Image')
#     axes[1].axis('off')

#     plt.tight_layout()  # Adjust layout to prevent overlapping titles
#     plt.show()
print("All trainings finished.")