import torch
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from models.vae import VAE
from models.conditional import UNetPACA, ControlNet, DegradationRemovalModuleResnet
from models.diffusion import Diffusion
from utils.dataset import PreprocessedCBCTtoCTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cbct_slice_path = "../training_data/CBCT/REC-100/slice_28.png"
ct_slice_path = "../training_data/CT/volume-100/slice_28.png"
cbct_slice = Image.open(cbct_slice_path).convert('L')
ct_slice = Image.open(ct_slice_path).convert('L')

# Transform to match training
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Pad((0, 64, 0, 64)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])

])

CBCT = transform(cbct_slice).unsqueeze(0).to(device)
CT = transform(ct_slice).unsqueeze(0).to(device)

vae_path = "../pretrained_models/vae.pth"
unet_path = "../pretrained_models/unet.pth"
dr_path = "dr_results_2/dr_module.pth"
controlnet_path = "dr_results_2/controlnet.pth"
paca_path = "dr_results_2/unet_paca_layers.pth"
output_dir = "inference"
os.makedirs(output_dir, exist_ok=True)

vae = VAE().to(device) # Use your actual VAE init args if different
vae.load_state_dict(torch.load(vae_path, map_location=device))
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
print("- VAE loaded.")

dr_module = DegradationRemovalModuleResnet().to(device) # Use actual DR init args
dr_module.load_state_dict(torch.load(dr_path, map_location=device))
dr_module.eval()
for param in dr_module.parameters():
    param.requires_grad = False
print("- DR Module loaded.")

controlnet = ControlNet().to(device)
controlnet.load_state_dict(torch.load(controlnet_path, map_location=device))
controlnet.eval()
for param in controlnet.parameters():
    param.requires_grad = False
print("- ControlNet loaded.")

unet = UNetPACA().to(device)
unet.load_state_dict(torch.load(unet_path, map_location=device), strict=False)
unet.load_state_dict(torch.load(paca_path, map_location=device), strict=False)
print("- Base UNet and PACA weights loaded.")
unet.eval()
for param in unet.parameters():
    param.requires_grad = False

diffusion = Diffusion(device)

guidance_scales = [0.8, 1, 1.25, 1.5, 1.75, 2]

with torch.no_grad():
    for guidance_scale in guidance_scales:

        CBCT = CBCT.to(device)
        CT = CT.to(device)

        controlnet_input, _ = dr_module(CBCT)
        actual_control_features = controlnet(controlnet_input)

        # Determine Null condition for CFG
        # We need a tuple in the same structure as actual_control_features
        # For simplicity, using zero tensors with the same shape.
        # Run once to get shapes if needed (or use known shapes)
        projected_null = tuple(torch.zeros_like(t) for t in actual_control_features[0])
        zero_conv_null = tuple(torch.zeros_like(t) for t in actual_control_features[1])
        null_control_features = (projected_null, zero_conv_null)
        print("Conditioning features calculated.")
        print(f"Starting sampling with Guidance Scale (w={guidance_scale})...")
    
        # Initialize latent noise (shape based on VAE output)
        # Determine latent shape (e.g., [1, 4, 32, 32]) - run VAE once if needed
        latent_shape = vae.encode(torch.zeros(1, 1, 256, 256, device=device))[0].shape # Get shape from dummy encode
        z_t = torch.randn(latent_shape, device=device)
        T = diffusion.timesteps

        for t_int in tqdm(range(T - 1, -1, -1), desc=f"Sampling image", leave=False):
            t = torch.full((z_t.size(0),), t_int, device=device, dtype=torch.long)

            # --- CFG: Predict noise twice ---
            pred_noise_cond = unet(z_t, t, actual_control_features)
            pred_noise_uncond = unet(z_t, t, null_control_features)
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            # --- End CFG ---

            # --- DDPM Step ---
            beta_t = diffusion.beta[t_int].view(-1, 1, 1, 1)
            alpha_t = diffusion.alpha[t_int].view(-1, 1, 1, 1)
            alpha_cumprod_t = diffusion.alpha_cumprod[t_int].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
            sqrt_reciprocal_alpha_t = torch.sqrt(1.0 / alpha_t)

            model_mean_coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
            model_mean = sqrt_reciprocal_alpha_t * (z_t - model_mean_coef2 * pred_noise)

            if t_int > 0:
                variance = diffusion.beta[t_int].view(-1, 1, 1, 1) # Use posterior variance beta_t
                noise = torch.randn_like(z_t)
                z_t_minus_1 = model_mean + torch.sqrt(variance) * noise
            else:
                z_t_minus_1 = model_mean
            z_t = z_t_minus_1
            # --- End DDPM Step ---

        # --- Decode Final Latent ---
        z_0 = z_t
        generated_image = vae.decode(z_0)

        # --- Save Output ---
        print("Saving output image...")
        # Prepare images for saving (denormalize)
        generated_image_vis = (generated_image / 2 + 0.5).clamp(0, 1).squeeze(0) # Remove batch dim
        cbct_image_vis = (CBCT / 2 + 0.5).clamp(0, 1).squeeze(0) # Remove batch dim
        ct_image_vis = (CT / 2 + 0.5).clamp(0, 1).squeeze(0)

        images_to_save = [cbct_image_vis, generated_image_vis, ct_image_vis]
        output_filename = os.path.join(output_dir, f"output_cfg_{guidance_scale:.1f}.png")
        torchvision.utils.save_image(
            images_to_save,
            output_filename,
            nrow=len(images_to_save), # Arrange images horizontally
        )
        print(f"Saved comparison image to {output_filename}")
        print("Done.")