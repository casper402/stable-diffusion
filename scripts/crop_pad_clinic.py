from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os

# Crop box
x0, x1 = 85, 425
y0, y1 = 134, 372

# Pad & resize transform
transform = transforms.Compose([
    transforms.Pad((13, 64, 13, 64), fill=-1000),
    transforms.Resize((256, 256)),
])

# Helpers
def load_img(np_arr):
    return Image.fromarray(np_arr)

def crop(img):
    return img.crop((x0, y0, x1, y1))

def transform_img(img):
    return np.array(transform(img))

if __name__ == "__main__":
    in_dir = '/Users/Niklas/thesis/training_data/clinic'
    out_dir = '/Users/Niklas/thesis/training_data/clinic_cropped'
    os.makedirs(out_dir, exist_ok=True)

    for fn in os.listdir(in_dir):
        if not fn.lower().endswith('.npy'):
            continue

        in_path = os.path.join(in_dir, fn)
        # load
        np_img = np.load(in_path)

        # process
        img = load_img(np_img)
        img = crop(img)
        out = transform_img(img)

        # save under same name into clinic_cropped/
        out_path = os.path.join(out_dir, fn)
        np.save(out_path, out)

        print(f"  ✓ {fn} → saved to clinic_cropped/{fn}")
