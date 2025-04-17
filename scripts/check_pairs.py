import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from utils.dataset import PairedCTCBCTDatasetNPY

def check_pairs(manifest_path, split='train', transform=None, out_dir='pair_checks', n_samples=4):
    # 1) load dataset & sampler

    ds = PairedCTCBCTDatasetNPY(manifest_csv=manifest_path,
                                     split=split,
                                     transform=transform)
    loader = DataLoader(ds,
                        batch_size=n_samples,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False)
    
    # 2) grab one batch
    cbct_imgs, ct_imgs = next(iter(loader))  # [B,1,H,W]
    
    # 3) for each in the batch, save a side-by-side comparison
    os.makedirs(out_dir, exist_ok=True)
    for i in range(cbct_imgs.size(0)):
        # stack CBCT and CT horizontally
        pair = torch.cat([
            cbct_imgs[i].cpu(),
            ct_imgs[i].cpu()
        ], dim=2)  # now shape [1, H, 2W]
        
        save_path = os.path.join(out_dir, f'pair_{i+1}.png')
        save_image(pair, save_path, normalize=True)
        print(f"Saved {save_path}")
    
    print(f"\nWrote {cbct_imgs.size(0)} CBCT↔CT comparison images → {out_dir}/")

if __name__ == '__main__':
    import torchvision.transforms as transforms
    
    # match whatever you’ll use downstream
    transform = transforms.Compose([
        transforms.Pad((0, 64, 0, 64)),
        transforms.Resize((256, 256)),
    ])
    
    manifest_path = '../data_quick_loop/manifest.csv'
    check_pairs(manifest_path,
                split='train',
                transform=transform,
                out_dir='pair_checks/train',
                n_samples=4)
    
    # You can repeat for val/test:
    check_pairs(manifest_path, split='validation', out_dir='pair_checks/val', transform=transform)
    check_pairs(manifest_path, split='test',       out_dir='pair_checks/test', transform=transform)
