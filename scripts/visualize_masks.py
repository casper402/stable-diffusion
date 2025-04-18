import os
import random
import numpy as np
import matplotlib.pyplot as plt

def show_random_slice(ct_dir, mask1_dir, mask2_dir, max_volume, max_slice,
                      vmin_ct=-1000, vmax_ct=1000, overlay_alpha=0.3):
    """
    Pick a random existing CT slice and its liver/tumor masks, then display in a 4-panel figure.
    """
    # pick a random valid pair of files
    while True:
        vol_idx   = random.randint(0, max_volume)
        slice_idx = random.randint(0, max_slice)
        fname     = f"volume-{vol_idx}_slice_{slice_idx:03d}.npy"

        ct_path    = os.path.join(ct_dir,    fname)
        liver_path = os.path.join(mask1_dir, fname)
        tumor_path = os.path.join(mask2_dir, fname)

        if os.path.isfile(ct_path) and os.path.isfile(liver_path) and os.path.isfile(tumor_path):
            break

    print(f"Random pick → volume {vol_idx}, slice {slice_idx:03d}")

    # load the data
    ct = np.load(ct_path)
    lv = np.load(liver_path).astype(bool)
    tu = np.load(tumor_path).astype(bool)

    # count tumor pixels
    n_tumor = int(tu.sum())
    print(f"Tumor pixel count on this slice: {n_tumor}")

    # create the 4-panel figure
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 4))

    # 1) Liver mask
    ax0.imshow(lv, cmap="Reds", origin="lower", vmin=0, vmax=1,
               interpolation="none", aspect="equal")
    ax0.set_title(f"Liver Mask\n(v{vol_idx}, z={slice_idx:03d})")
    ax0.axis("off")

    # 2) Tumor mask
    ax1.imshow(tu, cmap="Blues", origin="lower", vmin=0, vmax=1,
               interpolation="none", aspect="equal")
    ax1.set_title(f"Tumor Mask (n={n_tumor})\n(v{vol_idx}, z={slice_idx:03d})")
    ax1.axis("off")

    # 3) CT only
    ax2.imshow(ct, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct,
               interpolation="none", aspect="equal")
    ax2.set_title(f"CT Only\n(v{vol_idx}, z={slice_idx:03d})")
    ax2.axis("off")

    # 4) CT + overlay
    ax3.imshow(ct, cmap="gray", origin="lower", vmin=vmin_ct, vmax=vmax_ct,
               interpolation="none", aspect="equal")
    ax3.imshow(np.ma.masked_where(~lv, lv), cmap="Reds", alpha=overlay_alpha,
               origin="lower", interpolation="none", aspect="equal", vmin=0, vmax=1)
    ax3.imshow(np.ma.masked_where(~tu, tu), cmap="Blues", alpha=overlay_alpha,
               origin="lower", interpolation="none", aspect="equal", vmin=0, vmax=1)
    ax3.set_title(f"CT + Overlay (n={n_tumor})\n(v{vol_idx}, z={slice_idx:03d})")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    # ─── CONFIG ─────────────────────────────────────────────────────
    ct_dir      = "/Users/Niklas/thesis/training_data/CT"
    mask1_dir   = "/Users/Niklas/thesis/training_data/masks/liver"
    mask2_dir   = "/Users/Niklas/thesis/training_data/masks/tumor"

    max_volume = 130  # volumes 0..130 inclusive
    max_slice  = 365  # slices 0..365 inclusive
    vmin_ct, vmax_ct = -1000, 1000
    overlay_alpha   = 0.3
    # ──────────────────────────────────────────────────────────────────

    try:
        while True:
            show_random_slice(ct_dir, mask1_dir, mask2_dir,
                              max_volume, max_slice,
                              vmin_ct, vmax_ct, overlay_alpha)
    except KeyboardInterrupt:
        print("\nExiting viewer.")

if __name__ == "__main__":
    main()
