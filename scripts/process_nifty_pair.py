#!/usr/bin/env python
import os
import numpy as np
import nibabel as nib
import concurrent.futures


# Just test volumes:
volumes = [3, 8, 12, 26, 32, 33, 35, 54, 59, 61, 106, 116, 129]

# All volumes
# volumes = [i for i in range(131)]

def process_nifti_pair(ct_folder, ct_fname, cbct_folder, cbct_fname, output_dir, debug=False):
    # --- 1) Load CT and CBCT (keep full CT range) ---
    ct_path   = os.path.join(ct_folder,   ct_fname)
    cbct_path = os.path.join(cbct_folder, cbct_fname)
    print(f"[Volume {ct_fname}] Loading CT:  {ct_path}")
    print(f"[Volume {ct_fname}] Loading CBCT: {cbct_path}")

    ct_img   = nib.load(ct_path)
    cbct_img = nib.load(cbct_path)
    ct_vol   = ct_img.get_fdata()       # full-spectrum HUs
    cbct_vol = cbct_img.get_fdata()     # raw CBCT intensities

    # --- 2) Random sampling to estimate scale & offset via least-squares ---
    flat_ct   = ct_vol.ravel()
    flat_cbct = cbct_vol.ravel()

    mask = np.isfinite(flat_ct) & np.isfinite(flat_cbct)
    valid_idxs = np.where(mask)[0]

    n_samples = min(20000, valid_idxs.size)
    sample_idxs = np.random.choice(valid_idxs, size=n_samples, replace=False)

    x = flat_cbct[sample_idxs]   # predictor (raw CBCT)
    y = flat_ct[sample_idxs]     # target    (true HU)

    a, b = np.polyfit(x, y, 1)
    scale, offset = a, b

    if debug:
        print(f"[Volume {ct_fname}] Fitted linear mapping: scale = {scale:.6f}, offset = {offset:.1f}")
        preds = scale*x + offset
        res   = y - preds
        print(f"[Volume {ct_fname}] Residuals: mean={res.mean():.1f}, std={res.std():.1f}")

    # --- 3) Apply mapping, clip to [-1000,1000] ---
    cbct_calib = cbct_vol * scale + offset
    cbct_calib = np.clip(cbct_calib, -1000, 1000)

    # --- 4) Rotate & save each slice as .npy in shared output_dir ---
    base = os.path.splitext(cbct_fname)[0]
    if base.startswith("REC-"):
        base = "volume-" + base[4:]

    for i in range(cbct_calib.shape[2]):
        slice_data   = cbct_calib[:, :, i]
        rotated_data = np.rot90(slice_data, k=-1)
        out_fname    = f"{base}_slice_{i:03d}.npy"
        np.save(os.path.join(output_dir, out_fname), rotated_data)
        if debug:
            print(f"[Volume {ct_fname}] Saved {out_fname}")

    print(f"[Volume {ct_fname}] Done; saved {cbct_calib.shape[2]} slices in {output_dir}")


def main():
    # Folders & parameters
    quality = 32
    ct_folder   = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT"
    cbct_folder = f"/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated/{quality}/3D"
    output_dir  = f"/Users/Niklas/thesis/training_data/CBCT/{quality}/test"
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    num_volumes = 1
    max_workers = min(8, (os.cpu_count() or 1))

    def worker(volume_idx):
        ct_fname   = f"volume-{volume_idx}.nii"
        cbct_fname = f"REC-{volume_idx}.nii"
        # Save all slices directly into output_dir
        process_nifti_pair(ct_folder, ct_fname, cbct_folder, cbct_fname, output_dir, debug=False)
        return volume_idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, idx): idx for idx in volumes}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                future.result()
                print(f"[Main] Volume {idx} completed successfully.")
            except Exception as e:
                print(f"[Main] Volume {idx} failed with error: {e}")


if __name__ == "__main__":
    main()

