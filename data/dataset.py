import os
from torch.utils.data import Dataset, DataLoader
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
    
def get_dataloaders(config, train_transform, val_transform):
    train_dataset = CBCTtoCTDataset(config["paths"]["train_CBCT"], config["paths"]["train_CT"], config["model"]["image_size"], train_transform)
    val_dataset = CBCTtoCTDataset(config["paths"]["val_CBCT"], config["paths"]["val_CT"], config["model"]["image_size"], val_transform)

    train_dataloader = DataLoader(train_dataset, 
                            batch_size=config["train"]["batch_size"], 
                            shuffle=True, 
                            sampler=None, 
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    val_dataloader = DataLoader(val_dataset,
                            batch_size=config["train"]["batch_size"],
                            shuffle=False,
                            sampler=None,
                            num_workers=config["train"]["num_workers"],
                            pin_memory=True,
                            drop_last=True)
    
    return train_dataloader, val_dataloader