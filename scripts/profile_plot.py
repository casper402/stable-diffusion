import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import profile_line
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random

# ──────── constants & transforms ─────────────────────────────────────────────
DATA_RANGE = 2000.0    # CT range -1000…1000
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))

# Slice ranges per volume (inclusive); None means use all slices
SLICE_RANGES = {
    3: None, 8: (0, 354), 12: (0, 320), 26: None,
    32: (69, 269), 33: (59, 249), 35: (91, 268),
    54: (0, 330), 59: (0, 311), 61: (0, 315),
    106: None, 116: None, 129: (5, 346)
}

# Transform for CT/CBCT (pad + resize)
gt_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W), interpolation=InterpolationMode.BILINEAR),
])

def apply_transform(img_np: np.ndarray) -> np.ndarray:
    """
    Pad and resize CT/CBCT slice.
    """
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    return gt_transform(t).squeeze(0).numpy()


def crop_back(arr: np.ndarray) -> np.ndarray:
    """
    Crop padded/resized image back to original aspect.
    """
    return arr[TOP_CROP:RES_H-BOTTOM_CROP, LEFT_CROP:RES_W-RIGHT_CROP]


def load_and_prepare(path: str, is_cbct: bool = True) -> np.ndarray:
    """
    Load a .npy slice and apply transforms for CBCT/CT or crop only for predictions.
    """
    data = np.load(path)
    if is_cbct:
        data = apply_transform(data)
    return crop_back(data)


def list_slices(volume: int, folder: str) -> list:
    """
    Return list of slice filenames for a given volume, applying SLICE_RANGES.
    """
    pattern = os.path.join(folder, f"volume-{volume}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    rng = SLICE_RANGES.get(volume)
    if rng is not None:
        start, end = rng
        filtered = []
        for fp in files:
            idx = int(os.path.basename(fp).split('_')[-1].split('.')[0])
            if start <= idx <= end:
                filtered.append(fp)
        files = filtered
    return [os.path.basename(f) for f in files]


def plot_profile_multi(
    slice_name: str,
    x_coord: int,
    gt_folder: str,
    cbct_folder: str,
    pred_folders: dict,
    volume: int
):
    """
    Display CT slice with a vertical line and HU profiles for CT, CBCT, and multiple predictions.

    pred_folders: dict mapping label->folder containing volume subfolder.
    """
    # Build paths and load images
    ct_img = load_and_prepare(os.path.join(gt_folder, slice_name), is_cbct=True)
    cbct_img = load_and_prepare(os.path.join(cbct_folder, slice_name), is_cbct=True)

    preds = {}
    for label, folder in pred_folders.items():
        pred_path = os.path.join(folder, f"volume-{volume}", slice_name)
        preds[label] = load_and_prepare(pred_path, is_cbct=False)

    # Determine x-coordinate (random if None)
    H, W = ct_img.shape
    if x_coord is None:
        x_coord = random.randint(0, W-1)
    start, end = (0, x_coord), (H-1, x_coord)

    # Sample intensity profiles
    profs = {'CT': profile_line(ct_img, start, end, mode='constant', cval=np.nan),
             'CBCT': profile_line(cbct_img, start, end, mode='constant', cval=np.nan)}
    for label, img in preds.items():
        profs[label] = profile_line(img, start, end, mode='constant', cval=np.nan)

    x = np.arange(len(profs['CT']))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(ct_img, cmap='gray', vmin=-1000, vmax=1000)
    ax1.axvline(x=x_coord, color='r', lw=2)
    ax1.set_title(f'Volume {volume}, {slice_name}')
    ax1.axis('off')

    for label, prof in profs.items():
        ax2.plot(x, prof, label=label)
    ax2.set_xlabel('Pixel position (top to bottom)')
    ax2.set_ylabel('Hounsfield Unit (HU)')
    ax2.set_title(f'HU Profile at x={x_coord}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/test")
    pred_folders = {
        'v3': os.path.expanduser("~/thesis/predictions/predctions_controlnet_v3"),
        'v7': os.path.expanduser("~/thesis/predictions/predictions_controlnet_v7-data-augmentation")
    }

    volumes = list(SLICE_RANGES.keys())
    print(f"Using volumes: {volumes}")

    # Infinite loop: randomly pick volume, slice, x-coordinate
    while True:
        vol = random.choice(volumes)
        slices = list_slices(vol, gt_folder)
        if not slices:
            continue
        slice_name = random.choice(slices)

        # Set x_coord=None for random or provide an int
        plot_profile_multi(
            slice_name,
            x_coord=112,
            gt_folder=gt_folder,
            cbct_folder=cbct_folder,
            pred_folders=pred_folders,
            volume=vol
        )