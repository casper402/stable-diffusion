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
    def __init__(self, CBCT_path, CT_path, image_size, transform):
        self.CBCT_slices = self._collect_slices(CBCT_path)
        self.CT_slices = self._collect_slices(CT_path)
        self.image_size = image_size
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
        return min(len(self.CBCT_slices), len(self.CT_slices))
    
    def __getitem__(self, idx):
        CBCT_path = self.CBCT_slices[idx]
        CT_path = self.CT_slices[idx]

        CBCT_slice = Image.open(CBCT_path).convert("L")
        CT_slice = Image.open(CT_path).convert("L")
        
        if self.transform:
            CBCT_slice = self.transform(CBCT_slice)
            CT_slice = self.transform(CT_slice)

        return CBCT_slice, CT_slice