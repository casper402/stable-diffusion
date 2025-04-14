import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 256
CT_PATH = "../../training_data/CT/slice_1.png"  # Update if needed

def preprocess_image_pil(path, size=IMG_SIZE):
    image = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    return image  # 1-channel

# Load VAE
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE, dtype=DTYPE)
vae.eval()

# Load and preprocess image
original_image = preprocess_image_pil(CT_PATH).to(DEVICE)
input_tensor = original_image.repeat(3, 1, 1).unsqueeze(0).to(DTYPE)

# Encode → Decode
with torch.no_grad():
    latents = vae.encode(input_tensor).latent_dist.sample() * 0.18215
    decoded = vae.decode(latents).sample
    decoded = (decoded.clamp(-1, 1) + 1) / 2
    decoded = decoded[0].cpu().permute(1, 2, 0).numpy()
    decoded_gray = decoded[:, :, 0]  # just one channel

# Prepare original for display
original_np = original_image.cpu().numpy()[0]

# Create side-by-side comparison
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(original_np, cmap="gray")
axs[0].set_title("Original CT")
axs[0].axis("off")

axs[1].imshow(decoded_gray, cmap="gray")
axs[1].set_title("VAE Reconstructed")
axs[1].axis("off")

# Save and show
plt.tight_layout()
output_path = "vae_comparison.png"
plt.savefig(output_path)
print(f"✅ Comparison saved to: {output_path}")

# Optional: show image if possible
try:
    plt.show()
except:
    print("⚠️ Could not open interactive window.")
