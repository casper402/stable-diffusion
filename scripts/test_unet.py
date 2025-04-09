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
from tqdm import tqdm

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
loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=4)

vae = VAE().to(device)
unet = UNet().to(device)
diffusion = Diffusion(device, timesteps=1000)

vae_path = "/home/casper/Documents/Thesis/pretrained_models/vae.pth"
unet_path = "/home/casper/Documents/Thesis/stable-diffusion/results/unet/v2/best_unet.pth"

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

def denoise(loader):
    with torch.no_grad():
        for CT in loader:
            CT = CT.to(device)

            z_mu, z_logvar = vae.encode(CT)
            z = vae.reparameterize(z_mu, z_logvar)

            print(z.shape)

            t = diffusion.sample_timesteps(z.shape[0])
            noise = torch.randn_like(z)
            z_noisy = diffusion.add_noise(z, t, noise)

            pred_noise = unet(z_noisy, t)

            loss = noise_loss(pred_noise, noise)

            # Approximate denoised latent
            alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

            z_denoised = (z_noisy - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t

            recon = vae.decode(z_denoised)

            test = vae.decode(z_noisy)

            input_img = CT[0].cpu().squeeze()
            recon_img = recon[0].cpu().squeeze()
            test_img = test[0].cpu().squeeze()
            fig, axs = plt.subplots(1, 3, figsize=(20, 10))

            axs[0].imshow(input_img, cmap='gray')
            axs[0].set_title("Original")
            axs[0].axis('off')

            axs[1].imshow(recon_img, cmap='gray')
            axs[1].set_title(
            f"UNet\nL1: {loss.item():.4f}\nt: {t[0]}"
            )
            axs[1].axis('off')

            axs[2].imshow(test_img, cmap='gray')
            axs[2].set_title("Noisy")
            axs[2].axis('off')
        
            plt.tight_layout()
            plt.show()

def generate_random_image(unet, vae, diffusion, device, latent_shape=(1, 4, 32, 32)):

    unet.eval()
    vae.eval()

    T = diffusion.timesteps

    with torch.no_grad():
        x_t = torch.randn(latent_shape, device=device)

        for t_int in tqdm(range(T - 1, -1, -1), desc="Sampling latent"): 
            t = torch.full((latent_shape[0],), t_int, device=device, dtype=torch.long)

            # --- Get diffusion parameters for current t ---
            # Ensure parameters are fetched correctly and shaped for broadcasting
            beta_t = diffusion.beta[t].view(-1, 1, 1, 1)
            alpha_t = diffusion.alpha[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = diffusion.alpha_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

            # --- Predict noise using the UNet ---
            pred_noise = unet(x_t, t)

            # --- Calculate x_{t-1} using the DDPM sampling formula ---
            # Calculate the mean of the distribution p(x_{t-1} | x_t)
            model_mean_coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
            model_mean = sqrt_recip_alpha_t * (x_t - model_mean_coef2 * pred_noise)

            # Add noise for stochasticity (unless t=0)
            if t_int > 0:
                # Calculate variance sigma_t^2. Using beta_t is a common choice.
                # See DDPM paper Appendix B for details on variance choices.
                variance = beta_t
                noise = torch.randn_like(x_t)
                # Sample x_{t-1}
                x_t_minus_1 = model_mean + torch.sqrt(variance) * noise
            else:
                # t = 0, the final step output is the mean
                x_t_minus_1 = model_mean

            # Update x_t for the next iteration of the loop
            x_t = x_t_minus_1

        # Loop finished, x_t now holds the final generated latent z_0
        z_0 = x_t

        # 3. Decode the generated latent representation using the VAE decoder
        generated_image_batch = vae.decode(z_0)

        generated_image = generated_image_batch[0].cpu()

        if generated_image.shape[0] == 1:
            plot_image = generated_image.squeeze(0)
            cmap = 'gray'
        elif generated_image.shape[0] == 3:
            plot_image = generated_image.permute(1, 2, 0)
            plot_image = torch.clip(plot_image, -1, 1)
            plot_image = (plot_image + 1) / 2
            cmap = None
        else:
            print(f"Warning: Unexpected number of channels ({generated_image.shape[0]}) for plotting.")
            plot_image = generated_image[0]
            cmap = 'gray'

        plt.imshow(plot_image.numpy(), cmap=cmap)
        plt.title("Generated Image (Latent Diffusion)")
        plt.axis('off')
        plt.show()

        return generated_image_batch[0].cpu()

if __name__ == "__main__":
    #denoise(loader)
    generate_random_image(unet, vae, diffusion, device)

