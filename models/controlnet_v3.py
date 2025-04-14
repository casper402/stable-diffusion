import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
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
# CBCT_DIR = '../training_data/CBCT' # grendel
# SCT_DIR = '../training_data/CT/volume-1' # grendel
CBCT_DIR = '../../training_data/CBCT' # limited local
SCT_DIR = '../../training_data/CT' # limited local
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 256
BATCH_SIZE = 2
NUM_EPOCHS = 1
LR = 1e-5
# ------------------------------------------------------

def preprocess_image_pil(path, size=None):
    """Unified CBCT/sCT image loader: grayscale → resized → [0,1] → 3-channel tensor"""
    if size is None:
        size = IMG_SIZE
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
        # self.filenames = sorted(os.listdir(cbct_dir))[:10] # Only 10 samples atm!
        self.filenames = sorted(os.listdir(cbct_dir))
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
    from paca import setup_models_with_paca, apply_paca_to_controlnet_output, save_paca_blocks, extract_unet_features

    dataset = CBCT2SCTDataset(CBCT_DIR, SCT_DIR, size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("📦 Loading models with PACA...")
    vae, unet, controlnet, scheduler, paca_blocks = setup_models_with_paca()
    vae.eval()
    unet.eval()
    controlnet.train()
    paca_blocks['down'].train()
    paca_blocks['mid'].train()

    # Optimizer for PACA + ControlNet only
    trainable_params = list(controlnet.parameters()) + list(paca_blocks['mid'].parameters()) + list(paca_blocks['down'].parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    mse_loss = nn.MSELoss()
    scaler = GradScaler()
    best_loss = float("inf")

    print(f"🔧 Training on {DEVICE}, PACA enabled, img size {IMG_SIZE}")

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for batch in dataloader:
            cbct = batch["conditioning_image"].to(DEVICE, dtype=DTYPE)
            sct = batch["target_image"].to(DEVICE, dtype=DTYPE)
            encoder_hidden_states = torch.zeros((cbct.size(0), 77, 768), device=DEVICE)

            with torch.no_grad():
                with autocast(dtype=DTYPE):
                    latents = vae.encode(sct).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (cbct.size(0),), device=DEVICE).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                controlnet_output = controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=cbct,
                )

                # Extract UNet features
                unet_features = extract_unet_features(unet, noisy_latents, timesteps, encoder_hidden_states)

                # Apply PACA
                down, mid = apply_paca_to_controlnet_output(unet_features, controlnet_output, paca_blocks)

                # Final prediction
                pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                ).sample

                loss = mse_loss(pred, noise)

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            else:
                print("⚠️ Skipping NaN/Inf loss")

        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            controlnet.save_pretrained(os.path.join(SAVE_DIR, "controlnet_best"))
            save_paca_blocks(paca_blocks, os.path.join(SAVE_DIR, "paca_best"))
            print(f"💾 Best model updated (loss: {best_loss:.6f})")

    print("🏁 Training complete.")

def infer(cbct_path, save_path="generated_sct.png"):
    from paca import PACA, apply_paca_to_controlnet_output, extract_unet_features, load_paca_blocks

    print("*** Loading models + PACA ***")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE, dtype=DTYPE)
    unet = UNet2DConditionModel.from_pretrained("trained_model/unet").to(DEVICE, dtype=DTYPE)
    controlnet = ControlNetModel.from_pretrained("trained_model/controlnet_best").to(DEVICE, dtype=DTYPE)
    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # Load PACA
    paca_blocks = {
        'down': nn.ModuleList([PACA(embed_dim=320).to(DEVICE) for _ in range(4)]),
        'mid': PACA(embed_dim=640).to(DEVICE)
    }
    load_paca_blocks(paca_blocks, os.path.join(SAVE_DIR, "paca_best"))
    for p in paca_blocks['down']:
        p.eval()
    paca_blocks['mid'].eval()

    vae.eval()
    unet.eval()
    controlnet.eval()

    print("*** Preprocessing CBCT ***")
    cbct = preprocess_image_pil(CBCT_DIR + "/" + cbct_path, size=IMG_SIZE).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    encoder_hidden_states = torch.zeros((1, 77, 768), device=DEVICE, dtype=DTYPE)

    latents = torch.randn((1, 4, IMG_SIZE // 8, IMG_SIZE // 8), device=DEVICE, dtype=DTYPE)
    scheduler.set_timesteps(50)

    for t in scheduler.timesteps:
        with torch.no_grad():
            controlnet_output = controlnet(latents, t, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cbct)
            unet_features = extract_unet_features(unet, latents, t, encoder_hidden_states)
            down, mid = apply_paca_to_controlnet_output(unet_features, controlnet_output, paca_blocks)
            noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states,
                              down_block_additional_residuals=down,
                              mid_block_additional_residual=mid).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        latents = latents / 0.18215
        image = vae.decode(latents).sample
        image = (image.clamp(-1, 1) + 1) / 2
        image = image[0].cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(save_path)
        print(f"✅ Inference complete! Image saved to: {save_path}")

        import matplotlib.pyplot as plt
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
