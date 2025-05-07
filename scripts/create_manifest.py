#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# -----------------------------------------
# Configuration: root data directory and output CSV path
DATA_ROOT      = os.path.join('..', 'training_data')  # relative root
OUTPUT_CSV     = os.path.join(DATA_ROOT, "manifest-filtered.csv")
MAE_THRESHOLD  = 300.0  # exclude pairs with MAE >= this for any CBCT variant
CBCT_VERSIONS  = ['256', '490']  # subfolders under CBCT
# -----------------------------------------

def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean absolute error between two arrays."""
    return np.nanmean(np.abs(a - b))


def create_manifest(data_root: str, output_csv: str):
    """
    Walks through CT, CBCT (multiple versions), liver, and tumor splits under data_root,
    pairs identically-named .npy slices, computes MAE for each CBCT variant,
    and writes out a manifest CSV including only pairs
    where all CBCT MAEs < MAE_THRESHOLD.
    Columns: [ct_path, cbct_256_path, cbct_490_path, liver_path, tumor_path, split]
    """
    splits = ['train', 'validation', 'test']
    records = []
    total_processed = 0

    for split in splits:
        ct_dir    = os.path.join(data_root, 'CT',      split)
        liver_dir = os.path.join(data_root, 'liver',   split)
        tumor_dir = os.path.join(data_root, 'tumor',   split)
        cbct_dirs = {ver: os.path.join(data_root, 'CBCT', ver, split)
                     for ver in CBCT_VERSIONS}

        ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.npy')]
        num_files = len(ct_files)
        print(f"Starting split '{split}' - {num_files} files to process")

        for idx, fn in enumerate(ct_files, start=1):
            ct_path    = os.path.join(ct_dir,    fn)
            liver_path = os.path.join(liver_dir, fn)
            tumor_path = os.path.join(tumor_dir, fn)
            cbct_paths = {ver: os.path.join(dirpath, fn)
                          for ver, dirpath in cbct_dirs.items()}

            # Ensure all files exist
            missing = [pth for pth in [ct_path, liver_path, tumor_path] + list(cbct_paths.values())
                       if not os.path.exists(pth)]
            if missing:
                raise FileNotFoundError(f"Missing file for {fn} in split '{split}': {missing}")

            # Load CT once and compute MAEs
            ct_img = np.load(ct_path)
            maes = {}
            for ver, cbct_path in cbct_paths.items():
                cbct_img = np.load(cbct_path)
                maes[ver] = compute_mae(ct_img, cbct_img)

            # Include only if all MAEs < threshold
            if all(m < MAE_THRESHOLD for m in maes.values()):
                record = {'ct_path': ct_path, 'liver_path': liver_path,
                          'tumor_path': tumor_path, 'split': split}
                for ver in CBCT_VERSIONS:
                    record[f'cbct_{ver}_path'] = cbct_paths[ver]
                records.append(record)

            # Log progress every 100 files
            if idx % 100 == 0 or idx == num_files:
                print(f"  Processed {idx}/{num_files} files in split '{split}'")
        total_processed += num_files
        print(f"Finished split '{split}'. Total records so far: {len(records)}\n")

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"Manifest written to {output_csv} ({len(df)} entries) out of {total_processed} processed files. Excluded CBCT MAE >= {MAE_THRESHOLD}.")


if __name__ == '__main__':
    create_manifest(DATA_ROOT, OUTPUT_CSV)
