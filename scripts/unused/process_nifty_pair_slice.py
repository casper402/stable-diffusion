#!/usr/bin/env python
import os
import numpy as np
import nibabel as nib
import concurrent.futures

def process_nifti_pair(ct_folder, ct_fname, cbct_folder, cbct_fname, output_dir, debug=False):
    # --- 1) Load CT and CBCT (keep full CT range) ---
    ct_path   = os.path.join(ct_folder,   ct_fname)
    cbct_path = os.path.join(cbct_folder, cbct_fname)
    ct_vol   = nib.load(ct_path).get_fdata()
    cbct_vol = nib.load(cbct_path).get_fdata()

    base = os.path.splitext(cbct_fname)[0]
    if base.startswith("REC-"):
        base = "volume-" + base[4:]

    # --- 2) Loop over slices and do per‐slice regression & save ---
    for i in range(cbct_vol.shape[2]):
        ct_slice   = ct_vol[:, :, i]
        cbct_slice = cbct_vol[:, :, i]

        # mask out NaNs/infs
        flat_ct   = ct_slice.ravel()
        flat_cbct = cbct_slice.ravel()
        mask = (
            np.isfinite(flat_ct) &
            np.isfinite(flat_cbct) &
            (flat_ct >= -1000) & (flat_ct <= 1000)
        )
        valid_idxs = np.where(mask)[0]

        # if there's nothing valid, skip
        if valid_idxs.size == 0:
            if debug:
                print(f"[Slice {i}] no valid voxels, skipping")
            continue

        # sample up to 20k pixels
        n_samples = min(20000, valid_idxs.size)
        sample_idxs = np.random.choice(valid_idxs, size=n_samples, replace=False)
        x = flat_cbct[sample_idxs]
        y = flat_ct[sample_idxs]

        # fit y = a*x + b
        a, b = np.polyfit(x, y, 1)
        if debug:
            preds = a*x + b
            res   = y - preds
            print(f"[Slice {i}] scale={a:.6f}, offset={b:.1f}, res mean={res.mean():.1f}, std={res.std():.1f}")

        # apply, clip, rotate & save
        slice_calib   = np.clip(cbct_slice * a + b, -1000, 1000)
        rotated_slice = np.rot90(slice_calib, k=-1)
        out_fname     = f"{base}_slice_{i:03d}.npy"
        np.save(os.path.join(output_dir, out_fname), rotated_slice)

        if debug:
            print(f"[Slice {i}] saved {out_fname}")

    print(f"[Volume {ct_fname}] Done; processed {cbct_vol.shape[2]} slices in {output_dir}")

def main():
    # Folders & parameters
    ct_folder   = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT"
    cbct_folder = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated/256"
    output_dir  = "/Users/Niklas/thesis/training_data/CBCT/scaledV4"
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    num_volumes = 11
    max_workers = min(8, (os.cpu_count() or 1))

    print(f"Processing concurrently with {max_workers} max workers")

    def worker(volume_idx):
        ct_fname   = f"volume-{volume_idx}.nii"
        cbct_fname = f"REC-{volume_idx}.nii"
        # Save all slices directly into output_dir
        process_nifti_pair(ct_folder, ct_fname, cbct_folder, cbct_fname, output_dir, debug=False)
        return volume_idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, idx): idx for idx in range(num_volumes)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                future.result()
                print(f"[Main] Volume {idx} completed successfully.")
            except Exception as e:
                print(f"[Main] Volume {idx} failed with error: {e}")


if __name__ == "__main__":
    main()
