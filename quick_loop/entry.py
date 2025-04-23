import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision

from utils.dataset import get_dataloaders, CTDatasetNPY, PairedCTCBCTDatasetNPY
from models.diffusion import Diffusion
from quick_loop.vae import load_vae, train_vae
from quick_loop.unet import load_unet, train_unet
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from quick_loop.unetControlPACA import load_unet_control_paca
from quick_loop.unetControlPACA import train_dr_control_paca

### CONFIG ###
train_size = None
val_size = None
test_size = None
batch_size = 4
num_workers = 4
epochs = 500
early_stopping = 50
patience = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "data_quick_loop"

vae_predict_dir = os.path.join(save_dir, "vae_predictions")
unet_predict_dir = os.path.join(save_dir, "unet_predictions")
conditional_predict_dir = os.path.join(save_dir, "conditional_predictions")

vae_save_path = os.path.join(save_dir, "vae.pth")
unet_save_path = os.path.join(save_dir, "unet.pth")
controlnet_save_path = os.path.join(save_dir, "controlnet.pth")
paca_layers_save_path = os.path.join(save_dir, "paca_layers.pth")
degradation_removal_save_path = os.path.join(save_dir, "degradation_removal.pth")

os.makedirs(save_dir, exist_ok=True)

manifest_path = "../data_quick_loop/manifest.csv"
train_loader, val_loader, _ = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=CTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size)

vae = load_vae(vae_save_path, trainable=False)
# train_vae(vae=vae, train_loader=train_loader, val_loader=val_loader, epochs=epochs, early_stopping=early_stopping, patience=patience, save_path=os.path.join(save_dir, "vae.pth"), predict_dir=vae_predict_dir)

unet = load_unet(unet_save_path, trainable=False)
# train_unet(unet=unet, vae=vae, train_loader=train_loader, val_loader=val_loader, epochs=epochs, early_stopping=early_stopping, patience=patience, save_path=os.path.join(save_dir, "unet.pth"), predict_dir=unet_predict_dir)

vae = load_vae(save_path=vae_save_path, trainable=False)
unet = load_unet_control_paca(unet_save_path=unet_save_path, paca_trainable=True)
controlnet = load_controlnet(save_path=unet_save_path, trainable=True)
dr_module = load_degradation_removal(trainable=True)
train_loader, val_loader, _ = get_dataloaders(manifest_path, batch_size=batch_size, num_workers=num_workers, dataset_class=PairedCTCBCTDatasetNPY, train_size=train_size, val_size=val_size, test_size=test_size)
train_dr_control_paca(vae=vae, unet=unet, controlnet=controlnet, dr_module=dr_module, train_loader=train_loader, val_loader=val_loader, epochs=epochs, save_dir=save_dir, predict_dir=conditional_predict_dir, early_stopping=early_stopping, patience=patience, epochs_between_prediction=50)

print("All trainings finished.")