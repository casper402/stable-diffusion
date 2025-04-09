import os
from torch.utils.data import Dataset
from PIL import Image

class CTDataset(Dataset):
    def __init__(self, CT_path, transform):
        self.CT_slices = self._collect_slices(CT_path)
        self.transform = transform
        
    def _collect_slices(self, dataset_path):
        slice_paths = []
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(subdir_path):
                for slice_name in os.listdir(subdir_path):
                    slice_path = os.path.join(subdir_path, slice_name)
                    slice_paths.append(slice_path)
        return slice_paths
    
    def __len__(self):
        return len(self.CT_slices)
    
    def __getitem__(self, idx):
        CT_path = self.CT_slices[idx]
        CT_slice = Image.open(CT_path).convert("L")
        if self.transform:
            CT_slice = self.transform(CT_slice)
        return CT_slice
    
class CBCTtoCTDataset(Dataset):
    def __init__(self, CBCT_path, CT_path, transform):
        self.CBCT_path = CBCT_path
        self.CT_path = CT_path
        self.transform = transform
        self.paired_slice_paths = self._find_paired_slices()

    def _find_paired_slices(self):
        paired_paths = []
        ct_volumes = sorted([
            d for d in os.listdir(self.CT_path)
            if os.path.isdir(os.path.join(self.CT_path, d)) and d.startswith('volume-')
        ])

        for ct_volume_name in ct_volumes:
            volume_num_str = ct_volume_name.split('-')[-1]
            cbct_rec_name = f"REC-{volume_num_str}"

            ct_volume_path = os.path.join(self.CT_path, ct_volume_name)
            # *** Assume corresponding CBCT directory exists ***
            cbct_rec_path = os.path.join(self.CBCT_path, cbct_rec_name)

            # List and sort slices within the CT volume for deterministic order
            ct_slices = sorted([
                f for f in os.listdir(ct_volume_path)
                if os.path.isfile(os.path.join(ct_volume_path, f)) # Simple file check is okay
            ])

            for slice_name in ct_slices:
                ct_slice_path = os.path.join(ct_volume_path, slice_name)
                # *** Construct corresponding CBCT slice path directly ***
                # *** Assume this file exists because of the guarantee ***
                cbct_slice_path = os.path.join(cbct_rec_path, slice_name)

                # Add the pair without checking os.path.isfile(cbct_slice_path)
                paired_paths.append((cbct_slice_path, ct_slice_path))
        return paired_paths
        
    def __len__(self):
        return len(self.paired_slice_paths)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        cbct_slice_path, ct_slice_path = self.paired_slice_paths[idx]

        cbct_slice = Image.open(cbct_slice_path).convert('L')
        ct_slice = Image.open(ct_slice_path).convert('L')

        if self.transform:
            cbct_slice = self.transform(cbct_slice)
            ct_slice = self.transform(ct_slice)

        return cbct_slice, ct_slice