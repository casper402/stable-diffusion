import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision

from utils.dataset import get_dataloaders, CTDatasetNPY, PairedCTCBCTDatasetNPY
from models.diffusion import Diffusion
from quick_loop.vae import load_vae, train_vae
from quick_loop.unet import load_unet, train_unet, train_unet_v2, train_joint, train_joint_v2
from quick_loop.unetConditional import load_cond_unet, train_cond_unet
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from quick_loop.unetControlPACA import load_unet_control_paca, train_dr_control_paca, train_dr_control_paca_v2,test_dr_control_paca

### CONFIG ###
train_size = None
val_size = None
test_size = 10
batch_size = 8
accumulation_steps = 1 # Effectively increases batch size to batch_size * accumulation_steps
num_workers = 8
epochs = 2000
early_stopping = 50
patience = 15
epochs_between_prediction = 5
base_channels = 256
dropout_rate = 0.1
learning_rate = 1e-4
warmup_lr = 1e-8
warmup_epochs = 5

# Load pretrained model paths
# load_dir = "controlnet_v3"
# load_dir = "controlnet_from_unet_trained_after_joint_round2"
# load_dir = "controlnet_v6"
# load_dir = "controlnet_v8-data-augmentation"
load_dir = "controlnet_v8-data-augmentation"
load_vae_path = os.path.join(load_dir, "vae_joint_vae.pth")
load_unet_path = os.path.join(load_dir, "unet_joint_unet.pth")
load_controlnet_path = os.path.join(load_dir, "controlnet.pth")
load_paca_layers_path = os.path.join(load_dir, "paca_layers.pth")
load_degradation_removal_path = os.path.join(load_dir, "dr_module.pth")

# Save prediction / model directories
# save_dir = "controlnet_v2"
# save_dir = "unet_with_decoding_loss_perceptual_lower_v2"
# save_dir = "train_from_joint"
# save_dir = "controlnet_v3"
# save_dir = "controlnet_from_unet_trained_after_joint"
# save_dir = "controlnet_v4"
# save_dir = "controlnet_from_unet_trained_after_joint_round2"
# save_dir = "controlnet_v7-data-augmentation"
save_dir = "controlnet_v11-data-augmentation"

os.makedirs(save_dir, exist_ok=True)
vae_predict_dir = os.path.join(save_dir, "vae_predictions")
unet_predict_dir = os.path.join(save_dir, "unet_predictions")
conditional_predict_dir = os.path.join(save_dir, "conditional_predictions")
vae_save_path = os.path.join(save_dir, "vae_joint_vae.pth")
unet_save_path = os.path.join(save_dir, "unet_joint_unet.pth")
controlnet_save_path = os.path.join(save_dir, "controlnet.pth")
paca_layers_save_path = os.path.join(save_dir, "paca_layers.pth")
degradation_removal_save_path = os.path.join(save_dir, "dr_module.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
manifest_no_cbct_path = "../manifest-full.csv" # without CBCT
# manifest_path = "../manifest-cbct.csv" # with CBCT
# manifest_path = "../data_quick_loop/manifest.csv" # Local config
manifest_path = "../training_data/manifest-filtered.csv"

# Augmentation
augmentation = {
    'degrees': (-1, 1),
    'translate': (0.10, 0.10),
    'scale': (0.10, 1.10),
    'shear': None,
}


# vae = load_vae(vae_save_path, trainable=False)

# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=CTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# train_vae(vae=vae, train_loader=train_loader, val_loader=val_loader, epochs=epochs, early_stopping=early_stopping, patience=patience, save_path=vae_save_path, predict_dir=vae_predict_dir)

# Unet v1
# unet = load_unet(save_path = unet_save_path, trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)
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

# Unet v2
# train_loader, val_loader, test_loader = get_dataloaders(manifest_no_cbct_path, batch_size=batch_size, num_workers=num_workers, dataset_class=CTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# vae = load_vae(load_vae_path, trainable=False)
# unet = load_unet(save_path=load_unet_path, trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)
# train_unet_v2(unet=unet, 
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
#            perceptual_loss=True,
# )

# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)

# vae = load_vae(save_path=vae_save_path, trainable=True)
# unet = load_unet(save_path=unet_save_path, trainable=True, base_channels=base_channels, dropout_rate=dropout_rate)

# Define your VAE‐loss weights:
# vae_loss_weights = {
#     'perceptual': 0.1,
#     'ssim':       0.9,
#     'mse':        0.0,
#     'kl':         1e-5,
#     'l1':         1.0,
# }

# Jointly train UNet + VAE
# train_joint_v2(
#     unet=unet,
#     vae=vae,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     test_loader=test_loader,
#     epochs=epochs,
#     save_unet_path=unet_save_path,
#     save_vae_path=vae_save_path,
#     learning_rate=5e-6,
#     weight_decay=1e-4,
#     gradient_clip_val=1.0,
#     early_stopping=early_stopping,
#     vae_loss_weights=vae_loss_weights,
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

# IF YOU HAVE TO TRAIN FROM "SCRATCH", THEN REMEMBER TO ADJUST THE LOAD PATHS ACCORDINGLY
# in particular, controlnet uses unet, unet has no paca lyaers and dr module is also blank
vae = load_vae(save_path=load_vae_path, trainable=False)
unet = load_unet_control_paca(unet_save_path=load_unet_path, paca_save_path=load_paca_layers_path, paca_trainable=True)
controlnet = load_controlnet(save_path=load_controlnet_path, trainable=True)
dr_module = load_degradation_removal(save_path=load_degradation_removal_path, trainable=True)

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
    epochs_between_prediction=10)

# Controlnet v2
# vae = load_vae(save_path=vae_save_path, trainable=False)
# unet = load_unet_control_paca(unet_save_path=unet_save_path, paca_trainable=True)
# controlnet = load_controlnet(save_path=unet_save_path, trainable=True)
# dr_module = load_degradation_removal(trainable=True)
# unet = load_unet_control_paca(unet_save_path=unet_save_path, paca_trainable=True)
# train_loader, val_loader, test_loader = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size, augmentation=augmentation)
# train_dr_control_paca_v2(
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
#     epochs_between_prediction=10)

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