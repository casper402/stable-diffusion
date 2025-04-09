import torch
from models.vae import VAE 
from utils.dataset import CBCTtoCTDataset
from utils.config import load_config, get_device
import matplotlib.pyplot as plt
from utils.losses import PerceptualLoss, SsimLoss
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.diffusion import Diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CBCTtoCTDataset('../training_data/CBCT','../training_data/CT', transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad((0, 64, 0, 64)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]))

subset_size = 10
subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=4)

vae = VAE().to(device)
checkpoint_path = "../pretrained_models/vae.pth"
vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
vae.eval()

with torch.no_grad():
    for CBCT, CT in loader:
        CBCT = CBCT.to(device)
        CT = CT.to(device)
        
        z_cbct, mu_cbct, logvar_cbct, recon_cbct = vae(CBCT)
        z_ct, mu_ct, logvar_ct, recon_ct = vae(CT)

        ct_img = CT[0].cpu().squeeze()
        cbct_img = CBCT[0].cpu().squeeze()
        cbct_recon_img = recon_cbct[0].cpu().squeeze()
        ct_recon_img = recon_ct[0].cpu().squeeze()
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))

        axs[0].imshow(ct_img, cmap='gray')

        axs[1].imshow(ct_recon_img, cmap='gray')

        axs[2].imshow(cbct_img, cmap='gray')

        axs[3].imshow(cbct_recon_img, cmap='gray')


        plt.tight_layout()
        plt.show()




