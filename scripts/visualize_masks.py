import os
import itertools
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_and_split_mask(mask_path, ct_shape):
    """
    Loads a NIfTI mask with shape (X, Y, C=2, 1, Z), drops the dummy axis,
    and returns two 3D boolean arrays: liver and tumor masks aligned to ct_shape.
    """
    data = nib.load(mask_path).get_fdata()
    data = np.squeeze(data)

    if data.ndim == 4:
        liver = data[:, :, 0, :] > 0
        tumor = data[:, :, 1, :] > 0
    else:
        raise ValueError(f"Expected a 4D mask of shape (X,Y,2,Z), got {data.shape}")

    # Align to CT shape if needed
    for name, arr in (('liver', liver), ('tumor', tumor)):
        if arr.shape != ct_shape:
            for perm in itertools.permutations(range(3)):
                if arr.transpose(perm).shape == ct_shape:
                    if name == 'liver':
                        liver = arr.transpose(perm)
                    else:
                        tumor = arr.transpose(perm)
                    break
            else:
                raise ValueError(f"Could not align {name} mask shape {arr.shape} to CT shape {ct_shape}")

    return liver.astype(bool), tumor.astype(bool)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ct_dir     = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT/"
mask_dir   = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINMasksAlignedToCBCT/masks01/"
min_vol    = 0          # first volume index
max_vol    = 130        # last volume index
# ────────────────────────────────────────────────────────────────────────────────

# Display settings
vmin_ct, vmax_ct     = -1000, 1000        # CT intensity range
overlay_alpha        = 0.2               # transparency for overlays

for volume_idx in range(min_vol, max_vol + 1):
    t_ct   = os.path.join(ct_dir,   f"volume-{volume_idx}.nii")
    t_mask = os.path.join(mask_dir, f"{volume_idx}.nii")
    if not os.path.exists(t_ct) or not os.path.exists(t_mask):
        continue

    # Load CT and clip
    ct_vol   = nib.load(t_ct).get_fdata(dtype=float)
    ct_vol   = np.clip(ct_vol, vmin_ct, vmax_ct)
    ct_shape = ct_vol.shape

    # Load masks
    liver_mask, tumor_mask = load_and_split_mask(t_mask, ct_shape)

    # Compute slice-wise tumor area
    z_dim        = ct_shape[2]
    tumor_counts = [int(np.sum(tumor_mask[:, :, z])) for z in range(z_dim)]

    # Identify slice with maximum tumor pixels
    if sum(tumor_counts) == 0:
        print(f"Volume {volume_idx}: no tumor slices.")
        continue
    best_z = int(np.argmax(tumor_counts))  # index of slice with most tumor

    # Count liver & tumor slices
    slices_l   = sum(1 for z in range(z_dim) if tumor_counts[z] >= 0 and np.any(liver_mask[:, :, z]))
    slices_t   = sum(1 for z in range(z_dim) if tumor_counts[z] > 0)
    print(f"Volume {volume_idx}: liver slices = {slices_l}, tumor slices = {slices_t}, showing slice {best_z} (tumor pixels = {tumor_counts[best_z]})")

    # Prepare slice data
    ct_slice    = ct_vol[:, :, best_z].T
    lv          = liver_mask[:, :, best_z].T
    tu          = tumor_mask[:, :, best_z].T
    count       = tumor_counts[best_z]

    # Plot single row of panels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(lv,  cmap="Reds",   origin="lower", vmin=0, vmax=1)
    ax0.set_title(f"Liver Mask\n(vol {volume_idx}, z={best_z})"); ax0.axis("off")

    ax1.imshow(tu,  cmap="Blues",  origin="lower", vmin=0, vmax=1)
    ax1.set_title(f"Tumor Mask (n={count})\n(vol {volume_idx}, z={best_z})"); ax1.axis("off")

    ax2.imshow(ct_slice, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct)
    ax2.set_title(f"CT Only\n(vol {volume_idx}, z={best_z})"); ax2.axis("off")

    # Overlay only on mask regions
    ax3.imshow(ct_slice, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct)
    ax3.imshow(np.ma.masked_where(~lv, lv),   cmap="Reds",   alpha=overlay_alpha, origin="lower", vmin=0, vmax=1)
    ax3.imshow(np.ma.masked_where(~tu, tu),   cmap="Blues",  alpha=overlay_alpha, origin="lower", vmin=0, vmax=1)
    ax3.set_title(f"CT + Overlay (n={count})\n(vol {volume_idx}, z={best_z})"); ax3.axis("off")

    plt.tight_layout()
    plt.show()
