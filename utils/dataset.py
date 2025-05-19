import os
import random
import numpy as np
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode


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
    
class PreprocessedCBCTtoCTDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.manifest_path = manifest_path
        self.transform = transform
        self.paired_slice_paths = self._load_manifest()

        if not self.paired_slice_paths:
             print(f"Manifest file '{manifest_path}' loaded 0 pairs. Check the file or its generation process.")
        else:
             print(f"Loaded {len(self.paired_slice_paths)} pairs from {manifest_path}")


    def _load_manifest(self):
        """Loads the paired paths from the CSV manifest file."""
        paired_paths = []
        try:
            with open(self.manifest_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)

                for i, row in enumerate(reader):
                    if len(row) == 2:
                        cbct_path, ct_path = row[0], row[1]
                        if not cbct_path or not ct_path:
                             print.warning(f"Empty path found in manifest row {i+2}. Skipping.")
                             continue
                        paired_paths.append((cbct_path, ct_path))
                    else:
                        print.warning(f"Skipping malformed row {i+2} in manifest: {row}")

        except FileNotFoundError:
            print(f"Manifest file not found: {self.manifest_path}")
        except Exception as e:
            print(f"Error reading manifest file {self.manifest_path}: {e}")

        return paired_paths

    def __len__(self):
        return len(self.paired_slice_paths)

    def __getitem__(self, idx):
        if idx >= len(self):
             raise IndexError("Index out of range")

        # Get paths from the pre-loaded list
        cbct_slice_path, ct_slice_path = self.paired_slice_paths[idx]

        try:
            # Load images
            cbct_slice = Image.open(cbct_slice_path).convert('L')
            ct_slice = Image.open(ct_slice_path).convert('L')

            # Apply transforms (individually, as in your last example)
            if self.transform:
                # Ensure your transform can handle single images if applied like this
                cbct_slice = self.transform(cbct_slice)
                ct_slice = self.transform(ct_slice)

            return cbct_slice, ct_slice

        except FileNotFoundError:
            print.error(f"File not found (path from manifest invalid?): {cbct_slice_path} or {ct_slice_path}")
            return None
        except Exception as e:
            print.error(f"Error loading or processing index {idx} ({cbct_slice_path}, {ct_slice_path}): {e}")
            return None

class CTDatasetNPY(Dataset):
    def __init__(self, manifest_csv: str, split: str, augmentation=None, preprocess="linear"):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.base_transform = transforms.Compose([
            transforms.Pad((0, 64, 0, 64), fill=-1),
            transforms.Resize((256, 256)),
        ])
        if augmentation != None:
            degrees = augmentation.get('degrees', 0)
            translate = augmentation.get('translate', None)
            scale = augmentation.get('scale', None)
            shear = augmentation.get('shear', None)
            self.augmentation_transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=degrees,       # Random rotation between -rot..+rot degrees
                    translate=translate, # Random translation up to fraction% horizontally and vertically
                    scale=scale, # Add slight scaling
                    shear=shear,           # Add slight shear
                    fill=-1              # Fill new pixels with -1, consistent with Pad
                ),
            ])
        else:
            self.augmentation_transform = None

        assert preprocess in ["linear", "tanh"]
        print("Using preprocessing:", preprocess)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ct = np.load(row['ct_path']).astype(np.float32)
        if preprocess == "linear":
            ct /= 1000.0
        elif preprocess == "tanh":
            # apply a "soft window" around ±150 HU:
            #  - inside ±150 HU it's almost linear (tanh(x/150) ≈ x/150 for |x|≲100),
            #  - beyond ±150 HU things smoothly compress toward ±1.
            ct = np.tanh(ct / 150.0)
        ct = torch.from_numpy(ct).unsqueeze(0)
        ct = self.base_transform(ct)
        if self.augmentation_transform:
            ct = self.augmentation_transform(ct)
        return ct

class CTDatasetWithMeta(CTDatasetNPY):
    """
    Extends CTDatasetNPY to also return the volume index for each sample.
    """
    def __getitem__(self, idx):
        ct = super().__getitem__(idx)
        row = self.df.iloc[idx]
        filename = os.path.basename(row['ct_path'])  # e.g. "volume-59_slice_192.npy"
        volume_part = filename.split('_')[0]          # "volume-59"
        volume_idx = int(volume_part.split('-')[1])   # 59
        return ct, volume_idx
    
class PairedCTCBCTDatasetNPY(Dataset):
    def __init__(self, manifest_csv: str, split: str, augmentation=None, preprocess="linear"):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.base_transform = transforms.Compose([
            transforms.Pad((0, 64, 0, 64), fill=-1),
            transforms.Resize((256, 256)),
        ])
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ct = np.load(row['ct_path']).astype(np.float32) / 1000.0
        ct = torch.from_numpy(ct).unsqueeze(0)

        # randomly choose 256 vs 490 CBCT
        size = random.choice([256, 490])
        cbct_path = row[f'cbct_{size}_path']
        cbct = np.load(cbct_path).astype(np.float32) / 1000.0
        cbct = torch.from_numpy(cbct).unsqueeze(0)

        if self.base_transform:
            ct   = self.base_transform(ct)
            cbct = self.base_transform(cbct)

        if self.augmentation:
            img_size = F.get_image_size(ct)
            affine_params = transforms.RandomAffine.get_params(
                degrees = self.augmentation.get('degrees', 0),
                translate = self.augmentation.get('translate', None),
                scale_ranges = self.augmentation.get('scale', None),
                shears = self.augmentation.get('shear', None),
                img_size = img_size
            )
            ct = F.affine(ct, *affine_params, interpolation=InterpolationMode.BILINEAR, fill=-1)
            cbct = F.affine(cbct, *affine_params, interpolation=InterpolationMode.BILINEAR, fill=-1)

        return ct, cbct
    
class PairedCTCBCTSegmentationDatasetNPY(Dataset):
    def __init__(self, manifest_csv: str, split: str, augmentation=None, preprocess="linear"):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.base_transform = transforms.Compose([
            transforms.Pad((0, 64, 0, 64), fill=-1),
            transforms.Resize((256, 256)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Pad((0, 64, 0, 64), fill=0),
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST_EXACT),
        ])
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ct   = np.load(row['ct_path']).astype(np.float32)  / 1000.0
        cbct = np.load(row['cbct_490_path']).astype(np.float32) / 1000.0
        ct   = torch.from_numpy(ct).unsqueeze(0)
        cbct = torch.from_numpy(cbct).unsqueeze(0)
        
        liver_path = row.get('liver_path')
        if liver_path and os.path.exists(liver_path):
            liver = np.load(liver_path).astype(np.float32)
        else:
            liver = torch.zeros(1, ct.shape[0], dtype=torch.float32)

        tumor_path = row.get('tumor_path')
        if tumor_path and os.path.exists(tumor_path):
            tumor = np.load(tumor_path).astype(np.float32)
        else:
            tumor = torch.zeros(1, ct.shape[0], dtype=torch.float32)
        
        segmentation_map = liver - tumor

        tumor = torch.from_numpy(tumor).unsqueeze(0) # Add channel dim (C, H, W)
        liver = torch.from_numpy(liver).unsqueeze(0) # Add channel dim (C, H, W)
        segmentation_map = torch.from_numpy(segmentation_map).unsqueeze(0)

        if self.base_transform:
            ct   = self.base_transform(ct)
            cbct = self.base_transform(cbct)

        if self.mask_transform:
            liver = self.mask_transform(liver)
            tumor = self.mask_transform(tumor)
            segmentation_map = self.mask_transform(segmentation_map)


        if self.augmentation:
            img_size = F.get_image_size(ct)
            affine_params = transforms.RandomAffine.get_params(
                degrees = self.augmentation.get('degrees', 0),
                translate = self.augmentation.get('translate', None),
                scale_ranges = self.augmentation.get('scale', None),
                shears = self.augmentation.get('shear', None),
                img_size = img_size
            )
            ct = F.affine(ct, *affine_params, interpolation=transforms.InterpolationMode.BILINEAR, fill=-1)
            cbct = F.affine(cbct, *affine_params, interpolation=transforms.InterpolationMode.BILINEAR, fill=-1)
            segmentation_map = F.affine(segmentation_map, *affine_params, interpolation=transforms.InterpolationMode.NEAREST_EXACT, fill=0)
            liver = F.affine(liver, *affine_params, interpolation=transforms.InterpolationMode.NEAREST_EXACT, fill=0)
            tumor = F.affine(tumor, *affine_params, interpolation=transforms.InterpolationMode.NEAREST_EXACT, fill=0)

        return ct, cbct, segmentation_map, liver, tumor
    
class SegmentationMaskDatasetNPY(Dataset):
    def __init__(self, manifest_csv: str, split: str, augmentation=None, preprocess="linear"):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.mask_transform = transforms.Compose([
            transforms.Pad((0, 64, 0, 64), fill=0),
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST_EXACT),
        ])
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        liver = np.load(row['liver_path']).astype(np.float32)
        tumor = np.load(row['tumor_path']).astype(np.float32)

        segmentation_map = liver - tumor

        liver = torch.from_numpy(liver).unsqueeze(0)
        tumor = torch.from_numpy(tumor).unsqueeze(0)
        segmentation_map = torch.from_numpy(segmentation_map).unsqueeze(0)

        if self.mask_transform:
            liver = self.mask_transform(liver)
            tumor = self.mask_transform(tumor)
            segmentation_map = self.mask_transform(segmentation_map)

        if self.augmentation:
            img_size = F.get_image_size(liver)
            affine_params = transforms.RandomAffine.get_params(
                degrees=self.augmentation.get('degrees', 0),
                translate=self.augmentation.get('translate', None),
                scale_ranges=self.augmentation.get('scale', None),
                shears=self.augmentation.get('shear', None),
                img_size=img_size
            )

            segmentation_map = F.affine(segmentation_map, *affine_params, interpolation=InterpolationMode.NEAREST_EXACT, fill=0)
            liver = F.affine(liver, *affine_params, interpolation=InterpolationMode.NEAREST_EXACT, fill=0)
            tumor = F.affine(tumor, *affine_params, interpolation=InterpolationMode.NEAREST_EXACT, fill=0)

        return segmentation_map, liver, tumor
    
def get_dataloaders(manifest_csv, batch_size, num_workers, dataset_class=PairedCTCBCTDatasetNPY, shuffle_train=True, drop_last=True, train_size=None, val_size=None, test_size=None, augmentation=None, preprocess=None):
    train_dataset = dataset_class(manifest_csv=manifest_csv, split='train', augmentation=augmentation, preprocess=preprocess)
    val_dataset = dataset_class(manifest_csv=manifest_csv, split='validation', augmentation=None, preprocess=preprocess)
    test_dataset = dataset_class(manifest_csv=manifest_csv, split='test', augmentation=None, preprocess=preprocess)
    if train_size:
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
    if val_size:
        val_dataset, _ = random_split(val_dataset, [val_size, len(val_dataset) - val_size])
    if test_size:
        test_dataset, _ = random_split(test_dataset, [test_size, len(test_dataset) - test_size])

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Train loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )

    # Validation loader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Test loader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader