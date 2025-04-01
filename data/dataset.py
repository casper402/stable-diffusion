import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
    
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
    
class CTDataset(Dataset):
    def __init__(self, CT_path, image_size, transform):
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
        return len(self.CT_slices)
    
    def __getitem__(self, idx):
        CT_path = self.CT_slices[idx]
        CT_slice = Image.open(CT_path).convert("L")
        if self.transform:
            CT_slice = self.transform(CT_slice)
        return CT_slice
    
def get_ct_dataloaders(config, transform):
    full_dataset = CTDataset(
        CT_path=config["paths"]["train_CT"],
        image_size=config["model"]["image_size"],
        transform=transform
    )
    
    generator = torch.Generator().manual_seed(42)
    
    train_length = int(len(full_dataset)*0.8)
    val_length = len(full_dataset)-train_length
    train_dataset, val_dataset = random_split(full_dataset, [train_length, val_length], generator=generator)
    print(f"Splitting {len(full_dataset)} images into {len(train_dataset)} train images and {len(val_dataset)} validation images.")

    train_subset_length = int(config["train"]["subset_size"]*0.8)
    train_subset, _ = random_split(train_dataset, [train_subset_length, len(train_dataset) - train_subset_length])
    print(f"Random train subset of {len(train_subset)} images")

    val_subset_length = int(config["train"]["subset_size"]*0.2)
    val_subset, _ = random_split(val_dataset, [val_subset_length, len(val_dataset) - val_subset_length])
    print(f"Random val subset of {len(val_subset)} images")

    train_dataloader = DataLoader(train_subset, 
                            batch_size=config["train"]["batch_size"], 
                            shuffle=True, 
                            sampler=None, 
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    val_dataloader = DataLoader(val_subset,
                            batch_size=config["train"]["batch_size"],
                            shuffle=False,
                            sampler=None,
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    return train_dataloader, val_dataloader
    
def get_dataloaders(config, transform):
    full_dataset = CBCTtoCTDataset(
        CBCT_path=config["paths"]["train_CBCT"],
        CT_path=config["paths"]["train_CT"],
        image_size=config["model"]["image_size"],
        transform=transform
    )
    generator = torch.Generator().manual_seed(42)

    train_length = int(len(full_dataset)*0.8)
    val_length = len(full_dataset)-train_length
    train_dataset, val_dataset = random_split(full_dataset, [train_length, val_length], generator=generator)
    print(f"Splitting {len(full_dataset)} images into {len(train_dataset)} train images and {len(val_dataset)} validation images.")

    train_subset_length = int(config["train"]["subset_size"]*0.8)
    train_subset, _ = random_split(train_dataset, [train_subset_length, len(train_subset) - train_subset_length])
    print(f"Random train subset of {len(train_subset)} images")

    val_subset_length = int(config["train"]["subset_size"]*0.2)
    val_subset, _ = random_split(val_dataset, [val_subset_length, len(val_subset) - val_subset_length])
    print(f"Random val subset of {len(val_subset)} images")

    train_dataloader = DataLoader(train_subset, 
                            batch_size=config["train"]["batch_size"], 
                            shuffle=True, 
                            sampler=None, 
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    val_dataloader = DataLoader(val_subset,
                            batch_size=config["train"]["batch_size"],
                            shuffle=False,
                            sampler=None,
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    return train_dataloader, val_dataloader