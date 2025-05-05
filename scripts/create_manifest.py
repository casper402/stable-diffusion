import os
import pandas as pd

def create_manifest(data_root: str, output_csv: str):
    """
    Walks through CT_quick_loop and CBCT_quick_loop under data_root,
    pairs up identically‐named .npy slices in each split, and writes
    out a manifest CSV with columns [ct_path, cbct_path, split].
    """
    splits = ['train', 'validation', 'test']
    records = []

    for split in splits:
        ct_dir   = os.path.join(data_root, 'CT',  split)
        cbct_dir = os.path.join(data_root, 'CBCT', split)
        # Gather all CT slice names
        ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.npy')]
        
        for fn in ct_files:
            ct_path   = os.path.join(ct_dir,  fn)
            cbct_path = os.path.join(cbct_dir, fn)
            if not os.path.exists(cbct_path):
                raise FileNotFoundError(f"No matching CBCT for {fn} in {split}")
            records.append({
                'ct_path':   ct_path,
                'cbct_path': cbct_path,
                'split':     split
            })

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"Manifest written to {output_csv} ({len(df)} entries)")

if __name__ == '__main__':
    root = '../training_data'
    create_manifest(root, root + '/manifest-full.csv')
