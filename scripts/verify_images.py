import os
from PIL import Image
import torch
from torchvision import transforms

# ---------------- CONFIG ----------------
# CBCT_DIR = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated2D/256/REC-1'
# SCT_DIR = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT2D/volume-1'
CBCT_DIR = '../../training_data/CBCT'
SCT_DIR = '../../training_data/CT/volume-1'
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
    return image.repeat(3, 1, 1)

def is_valid_image(filename):
    return (
        filename.endswith(".png") and
        not filename.startswith("._") and
        not filename.startswith(".")
    )

def verify_images(cbct_dir, sct_dir, size=512):
    filenames = sorted([
        f for f in os.listdir(cbct_dir)
        if is_valid_image(f)
    ])

    if not filenames:
        print("❌ No valid CBCT images found.")
        return

    passed = 0
    failed = 0

    for filename in filenames:
        cbct_path = os.path.join(cbct_dir, filename)
        sct_path = os.path.join(sct_dir, filename)

        if not os.path.isfile(sct_path):
            print(f"⚠️ Missing sCT image for: {filename}")
            failed += 1
            continue

        try:
            cbct = preprocess_image_pil(cbct_path, size)
            sct = preprocess_image_pil(sct_path, size)

            assert cbct.shape == (3, size, size)
            assert sct.shape == (3, size, size)

            passed += 1
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")
            failed += 1

    print("\n✅ Check complete.")
    print(f"✔️ Passed: {passed}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    verify_images(CBCT_DIR, SCT_DIR, size=IMG_SIZE)
