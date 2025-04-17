import torch
from models.vae import VAE 
from utils.dataset import CTDatasetNPY
from utils.config import load_config, get_device
import matplotlib.pyplot as plt
from utils.losses import PerceptualLoss, SsimLoss
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.diffusion import Diffusion

from torchvision import transforms
from torchvision.transforms import InterpolationMode

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor_transform = transforms.Compose([
        transforms.Pad((0, 64, 0, 64)),                               # picklable
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),  # picklable
    ])

    dataset = CTDatasetNPY('../../training_data/CT', transform=tensor_transform, limit=10)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    vae = VAE().to(device)
    checkpoint_path = "../best_vae_ct_2.pth"
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vae.eval()

    with torch.no_grad():
        for CT in loader:
            CT = CT.to(device)
            
            z_ct, mu_ct, logvar_ct, recon_ct = vae(CT)

            ct_img = CT[0].cpu().squeeze()
            ct_recon_img = recon_ct[0].cpu().squeeze()
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))

            axs[0].imshow(ct_img, cmap='gray')

            axs[1].imshow(ct_recon_img, cmap='gray')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()