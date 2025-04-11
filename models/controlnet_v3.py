import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
)

# ----------------------- CONFIG -----------------------
SAVE_DIR = 'trained_model'
# CBCT_DIR = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated2D/256/REC-1' # full local
# SCT_DIR = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT2D/volume-1' # full local
CBCT_DIR = '../training_data/CBCT' # grendel
SCT_DIR = '../training_data/CT/volume-1' # grendel
# CBCT_DIR = '../../training_data/CBCT' # limited local
# SCT_DIR = '../../training_data/CT' # limited local
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 1e-5
# ------------------------------------------------------

def preprocess_image_pil(path, size=512):
    """Unified CBCT/sCT image loader: grayscale → resized → [0,1] → 3-channel tensor"""
    image = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # outputs 1 channel [C,H,W]
    ])
    image = transform(image)
    return image.repeat(3, 1, 1)  # make it 3-channel


class CBCT2SCTDataset(Dataset):
    def __init__(self, cbct_dir, sct_dir, size=512):
        self.cbct_dir = cbct_dir
        self.sct_dir = sct_dir
        self.filenames = sorted(os.listdir(cbct_dir))[:10] # Only 10 samples atm!
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        cbct = preprocess_image_pil(os.path.join(CBCT_DIR, filename))
        sct = preprocess_image_pil(os.path.join(SCT_DIR, filename))

        return {
            "conditioning_image": cbct,
            "target_image": sct
        }

# not actually used right now?
def load_models():
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE, dtype=DTYPE)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(DEVICE, dtype=DTYPE)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(DEVICE, dtype=DTYPE)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    return vae, unet, controlnet, scheduler

def train():
    dataset = CBCT2SCTDataset(CBCT_DIR, SCT_DIR, size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("loading models")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE, dtype=DTYPE)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(DEVICE, dtype=DTYPE)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(DEVICE, dtype=DTYPE)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    print("eval")
    vae.eval()

    print("train")
    unet.train()
    controlnet.train()

    optimizer = torch.optim.AdamW(list(unet.parameters()) + list(controlnet.parameters()), lr=LR)

    mse_loss = nn.MSELoss()
    best_loss = float("inf")  # TODO: testing — track best model

    print(f"🔧 Training on device: {DEVICE}, dtype: {DTYPE}")

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for batch in dataloader:
            cbct = batch["conditioning_image"].to(DEVICE, dtype=DTYPE)
            sct = batch["target_image"].to(DEVICE, dtype=DTYPE)

            # TODO: testing — check for bad input values
            if torch.isnan(cbct).any() or torch.isinf(cbct).any():
                print("⚠️ CBCT input contains NaNs or Infs — skipping batch")
                continue
            if torch.isnan(sct).any() or torch.isinf(sct).any():
                print("⚠️ SCT input contains NaNs or Infs — skipping batch")
                continue

            batch_size = cbct.shape[0]
            encoder_hidden_states = torch.zeros((batch_size, 77, 768), device=DEVICE, dtype=DTYPE)

            with torch.no_grad():
                latents = vae.encode(sct).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=DEVICE).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # TODO: testing — check latent tensor before forward
            if torch.isnan(noisy_latents).any():
                print("⚠️ Noisy latents contain NaNs — skipping batch")
                continue

            controlnet_output = controlnet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=cbct
            )

            pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=controlnet_output.down_block_res_samples,
                mid_block_additional_residual=controlnet_output.mid_block_res_sample
            ).sample

            loss = mse_loss(pred, noise)

            # TODO: testing — handle NaN loss
            if torch.isnan(loss):
                print("❌ Loss is NaN — skipping step")
                continue

            loss.backward()

            # TODO: testing — gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)

            print("stepping")
            optimizer.step()
            optimizer.zero_grad()


            print("loss added")
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

        # TODO: testing — save best model based on validation or lowest loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            unet.save_pretrained(os.path.join(SAVE_DIR, "unet_best"))
            controlnet.save_pretrained(os.path.join(SAVE_DIR, "controlnet_best"))
            print(f"💾 Best model updated! Loss: {best_loss:.6f}")

    print("🏁 Training complete.")

def infer(cbct_path, save_path="generated_sct.png"):
    # 🔁 Load trained models
    print("*** Loading models ***")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE, dtype=DTYPE)
    unet = UNet2DConditionModel.from_pretrained("trained_model/unet").to(DEVICE, dtype=DTYPE)
    controlnet = ControlNetModel.from_pretrained("trained_model/controlnet").to(DEVICE, dtype=DTYPE)
    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    print("*** Evaluating models ***")
    vae.eval()
    unet.eval()
    controlnet.eval()

    # 📥 Preprocess CBCT image
    print("*** Preprocessing CBCT ***")
    cbct = preprocess_image_pil(CBCT_DIR + "/" + cbct_path, size=IMG_SIZE).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    # Dummy prompt embedding (required for shape)
    print("*** Dummy prompt embedding ***")
    batch_size = cbct.shape[0]
    encoder_hidden_states = torch.zeros((batch_size, 77, 768), device=DEVICE, dtype=DTYPE)

    # 🌀 Start from pure noise in latent space
    print("*** Pure noice in latent space ***")
    latents = torch.randn((1, 4, IMG_SIZE // 8, IMG_SIZE // 8), device=DEVICE, dtype=DTYPE)
    scheduler.set_timesteps(50)

    # 🔁 Denoising loop
    print("*** Denoising loop ***")
    for t in scheduler.timesteps:
        print(f"--- Loop: {t}")
        with torch.no_grad():
            controlnet_output = controlnet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=cbct
            )
            down = controlnet_output.down_block_res_samples
            mid = controlnet_output.mid_block_res_sample

            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid
            ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 🎨 Decode latent to image
    print("*** Decoding latent to image ***")
    with torch.no_grad():
        latents = latents / 0.18215
        image = vae.decode(latents).sample
        image = (image.clamp(-1, 1) + 1) / 2
        image = image[0].cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(save_path)
        print(f"✅ Inference complete! Image saved to: {save_path}")

        # Optional preview
        plt.imshow(image[:, :, 0], cmap="gray")
        plt.title("Generated sCT")
        plt.axis("off")
        plt.show()

# ---- Choose mode: "train" or "infer" ----
if __name__ == "__main__":
    MODE = "train"
    if MODE == "train":
        train()
    elif MODE == "infer":
        infer("slice_10.png")
