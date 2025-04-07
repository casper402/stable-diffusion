import torch
from models.vae import VAE 
from models.unet import UNet
from utils.dataset import CTDataset
from utils.config import get_device
import matplotlib.pyplot as plt
from utils.losses import PerceptualLoss, SsimLoss
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.diffusion import Diffusion

def noise_loss(pred_noise, true_noise):
    return F.mse_loss(pred_noise, true_noise)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CTDataset('../training_data/CT', transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad((0, 64, 0, 64)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]))

subset_size = 10
subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=4)

vae = VAE().to(device)
unet = UNet().to(device)
diffusion = Diffusion(device, timesteps=1000)

vae_path = "/home/casper/Documents/Thesis/pretrained_models/vae.pth"
unet_path = "/home/casper/Documents/Thesis/stable-diffusion/results/unet/best_unet.pth"
try:
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    print(f"Loaded VAE weights from {vae_path}")
except Exception as e:
    print(f"Could not load VAE weights: {e}")
try:
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    print(f"Loaded UNet weights from {unet_path}")
except Exception as e:
    print(f"Could not load Unet weights: {e}")

vae.eval()
unet.eval()

with torch.no_grad():
    for CT in loader:
        CT = CT.to(device)

        z_mu, z_logvar = vae.encode(CT)
        z = vae.reparameterize(z_mu, z_logvar)

        t = diffusion.sample_timesteps(z.shape[0])
        noise = torch.randn_like(z)
        z_noisy = diffusion.add_noise(z, t, noise)

        pred_noise = unet(z_noisy, t)

        loss = noise_loss(pred_noise, noise)

        alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
        z_denoised = (z_noisy - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)

        recon = vae.decode(z_denoised)

        input_img = CT[0].cpu().squeeze()
        recon_img = recon[0].cpu().squeeze()
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        axs[0].imshow(input_img, cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(recon_img, cmap='gray')
        axs[1].set_title(
        f"Reconstruction\nL1: {loss.item():.4f}"
        )
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()





