import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision

from utils.dataset import get_dataloaders, CTDatasetNPY, PairedCTCBCTDatasetNPY, PairedCTCBCTSegmentationDatasetNPY, SegmentationMaskDatasetNPY
from models.diffusion import Diffusion
from quick_loop.vae import load_vae, train_vae
from quick_loop.unet import load_unet, train_unet, train_joint
from quick_loop.unetConditional import load_cond_unet, train_cond_unet, UNetConcatenation
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from quick_loop.unetControlPACA import load_unet_control_paca, train_dr_control_paca, test_dr_control_paca, train_segmentation_control
from quick_loop.unetConcatControlPACA import load_unet_concat_control_paca, train_unet_concat_control_paca

### CONFIG ###
train_size = None
val_size = None
test_size = 10
batch_size = 16
num_workers = 8
epochs = 2000
early_stopping = 50
patience = 10
epochs_between_prediction = 10
base_channels = 256
dropout_rate = 0.0
learning_rate = 5e-5
warmup_lr = 0
warmup_epochs = 0

# Augmentation
augmentation = {
    'degrees': (-1, 1),
    'translate': (0.10, 0.10),
    'scale': (0.90, 1.10),
    'shear': None,
}
#augmentation = None

# Vae Loss params
perceptual_weight=0.05
ssim_weight=1
mse_weight=1.0
kl_weight=0.000001
l1_weight=0

# Load pretrained model paths
load_dir = "../best_model_v7"
load_vae_path = os.path.join(load_dir, "vae_joint_vae.pth")
load_dir = "unet_concatenation"
load_unet_path = os.path.join(load_dir, "unet_v2.pth")
load_dr_module_path = os.path.join(load_dir, "dr_module.pth")
load_controlnet_path = os.path.join(load_dir, "controlnet.pth")
load_paca_layers_path = os.path.join(load_dir, "paca_layers.pth")

# Save prediction / model directories
save_dir = "unet_concat_control_paca"
os.makedirs(save_dir, exist_ok=True)
vae_predict_dir = os.path.join(save_dir, "vae_predictions")
unet_predict_dir = os.path.join(save_dir, "unet_predictions")
conditional_predict_dir = os.path.join(save_dir, "conditional_predictions")
vae_save_path = os.path.join(save_dir, "vae.pth")
unet_save_path = os.path.join(save_dir, "unet.pth")
controlnet_save_path = os.path.join(save_dir, "controlnet.pth")
paca_layers_save_path = os.path.join(save_dir, "paca_layers.pth")
dr_module_save_path = os.path.join(save_dir, "dr_module.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
manifest_path = "../training_data/manifest-filtered.csv"
# manifest_path = "../training_data/manifest-full.csv" # without CBCT

# --- VAE ---
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=CTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# vae = load_vae(load_vae_path, trainable=True)
# train_vae(
#     vae=vae, 
#     train_loader=train_loader, 
#     val_loader=val_loader, 
#     epochs=epochs, 
#     early_stopping=early_stopping, 
#     patience=patience, 
#     save_path=vae_save_path, 
#     predict_dir=vae_predict_dir,
#     perceptual_weight=perceptual_weight,
#     ssim_weight=ssim_weight,
#     mse_weight=mse_weight,
#     kl_weight=kl_weight,
#     l1_weight=l1_weight,
#     learning_rate=learning_rate
# )

# --- UNET ---
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=CTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# vae = load_vae(load_vae_path, trainable=False)
# unet = load_unet(load_unet_path, trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)
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
#            learning_rate=learning_rate,
#            warmup_lr=warmup_lr,
#            warmup_epochs=warmup_epochs
# )

# --- Joint UNET and VAE ---
# vae = load_vae(save_path=load_vae_path, trainable=True)
# unet = load_unet(trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=CTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# vae_loss_weights = {
#     'perceptual': perceptual_weight,
#     'ssim':       ssim_weight,
#     'mse':        mse_weight,
#     'kl':         kl_weight,
#     'l1':         l1_weight,
# }
# train_joint(
#     unet=unet,
#     vae=vae,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     test_loader=test_loader,
#     epochs=5,
#     save_unet_path=unet_save_path,
#     save_vae_path=vae_save_path,
#     learning_rate=5e-6,
#     weight_decay=1e-4,
#     gradient_clip_val=1.0,
#     early_stopping=early_stopping,
#     vae_loss_weights=vae_loss_weights,
# )

# --- Joint continue UNet training with locked VAE ---
# vae = load_vae(save_path=vae_save_path, trainable=False)
# unet = load_unet(save_path=unet_save_path, trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)
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
#            learning_rate=learning_rate,
# )

# --- Conditional Unet ---
# vae = load_vae(load_vae_path)
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# unet = load_cond_unet(trainable=True, base_channels=base_channels, dropout_rate=dropout_rate, unet_type=UNetConcatenation)
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

# --- ControlNet ---
# vae = load_vae(save_path=load_vae_path, trainable=False)
# unet = load_unet_control_paca(unet_save_path=load_unet_path, paca_trainable=True)
# controlnet = load_controlnet(save_path=load_unet_path, trainable=True)
# dr_module = load_degradation_removal(save_path=load_dr_module_path, trainable=True)
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
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
#     epochs_between_prediction=10
# )

# --- UNet Concatenation ControlNet ---
vae = load_vae(load_vae_path, trainable=False)
unet = load_unet_concat_control_paca(unet_save_path=load_unet_path, paca_trainable=True)
controlnet = load_controlnet(trainable=True)
dr_module = load_degradation_removal(trainable=True)
train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
train_dr_control_paca(
    vae=vae, 
    unet=unet, 
    controlnet=controlnet, 
    dr_module=dr_module, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    epochs=epochs, 
    save_dir=save_dir, 
    predict_dir=conditional_predict_dir, 
    early_stopping=early_stopping, 
    patience=patience, 
    epochs_between_prediction=10
)



# --- Segmentation ControlNet ---
# vae = load_vae(load_vae_path)
# unet = load_unet_control_paca(load_unet_path, load_paca_layers_path)
# controlnet_cbct = load_controlnet(load_controlnet_path)
# dr_module_cbct = load_degradation_removal(load_dr_module_path)
# controlnet_seg = load_controlnet(load_unet_path, True)
# dr_module_seg = load_degradation_removal(None, True)
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTSegmentationDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# train_segmentation_control(
#     vae=vae, 
#     unet=unet, 
#     controlnet_cbct=controlnet_cbct, 
#     dr_module_cbct=dr_module_cbct,
#     controlnet_seg=controlnet_seg, 
#     dr_module_seg=dr_module_seg,
#     train_loader=train_loader, 
#     val_loader=val_loader, 
#     test_loader=test_loader,
#     epochs=epochs, 
#     save_dir=save_dir, 
#     early_stopping=early_stopping, 
#     patience=patience, 
#     epochs_between_prediction=5,
#     learning_rate=learning_rate
# )

# --- Test ControlNet ---
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

print("All trainings finished.")