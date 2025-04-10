import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# ---------------- CONFIG ----------------
CBCT_DIR = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated2D/256/REC-1'
SCT_DIR = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT2D/volume-1'
IMG_SIZE = 512
# ----------------------------------------


def preprocess_image_pil(path, size=512):
    """Grayscale → Resize → Tensor [0,1] → 3-channel"""
    image = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image.repeat(3, 1, 1)  # [3, H, W]


def get_random_filename():
    filenames = sorted(os.listdir(CBCT_DIR))
    if not filenames:
        raise FileNotFoundError("No CBCT images found.")
    return random.choice(filenames)


def show_images(title, *imgs):
    num = len(imgs)
    fig, axs = plt.subplots(1, num, figsize=(5 * num, 5))

    # Make sure axs is iterable
    if num == 1:
        axs = [axs]

    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
        axs[i].imshow(img[:, :, 0], cmap="gray")
        axs[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = get_random_filename()
    print(f"🔍 Visualizing: {filename}")

    cbct_path = os.path.join(CBCT_DIR, filename)
    sct_path = os.path.join(SCT_DIR, filename)

    cbct = preprocess_image_pil(cbct_path, size=IMG_SIZE)
    sct = preprocess_image_pil(sct_path, size=IMG_SIZE)

    show_images(f"Preprocessed CBCT for inference: {filename}", cbct)
    show_images(f"Training Pair: CBCT (left), sCT (right): {filename}", cbct, sct)
