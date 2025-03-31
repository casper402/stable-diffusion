import torch
from models.vae import VAE 
from data.dataset import get_ct_dataloaders, get_dataloaders
from data.transforms import build_train_transform
from utils.config import load_config, get_device
import matplotlib.pyplot as plt
from utils.losses import PerceptualLoss
from piq import ssim
import torch.nn.functional as F

def main():
    device = get_device()
    config = load_config(device)
    
    transform = build_train_transform(config["model"]["image_size"])
    train_loader, val_loader = get_ct_dataloaders(config, transform)

    vae = VAE(latent_dim=config["model"]["latent_dim"]).to(device)
    checkpoint_path = "/home/casper/Documents/Thesis/stable-diffusion/checkpoints/ssim_percept_kl.pth"
    # checkpoint_path = "/home/casper/Documents/Thesis/stable-diffusion/pretrained_models/model.ckpt"

    try:
        vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded VAE weights from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load VAE weights: {e}")

    vae.eval()
    percept = PerceptualLoss(device)

    with torch.no_grad():
        for CT in train_loader:
            CT = CT.to(device)
            z, mu, sd, recon = vae(CT)

            ssim_loss = 1 - ssim(recon, CT)
            perceptual_loss = percept(recon, CT)
            l2_loss = F.mse_loss(recon, CT, reduction='mean')
            l1_loss = F.l1_loss(recon, CT, reduction='mean')

            input_img = CT[0].cpu().squeeze()
            recon_img = recon[0].cpu().squeeze()
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))

            axs[0].imshow(input_img, cmap='gray')
            axs[0].set_title("Original")
            axs[0].axis('off')

            axs[1].imshow(recon_img, cmap='gray')
            axs[1].set_title(
                f"Reconstruction\nL1: {l1_loss.item():.4f} | L2: {l2_loss.item():.4f}\nSSIM: {ssim_loss.item():.4f} | Perc: {perceptual_loss.item():.4f}"
            )
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()

    




