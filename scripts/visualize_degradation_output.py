import torch
from quick_loop.degradationRemoval import load_degradation_removal
from utils.dataset import get_dataloaders, CTDatasetNPY, PairedCTCBCTDatasetNPY
import torch.nn as nn
import os
import matplotlib.pyplot as plt

def visualize_degradation_outputs(model: nn.Module, conditioning_tensor: torch.Tensor, batch_idx: int, img_idx_in_batch: int):
    original_training_state = model.training
    model.eval()
    device = next(model.parameters()).device

    conditioning_tensor = conditioning_tensor.to(device)

    with torch.no_grad():
        embedding, intermediate_preds = model(conditioning_tensor)

    model.train(original_training_state)

    pred_128, pred_64 = intermediate_preds

    input_img_np = ((conditioning_tensor.squeeze().cpu().numpy() + 1) / 2.0).clip(0, 1)
    pred_128_np = ((pred_128.squeeze().cpu().numpy() + 1) / 2.0).clip(0, 1)
    pred_64_np = ((pred_64.squeeze().cpu().numpy() + 1) / 2.0).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(input_img_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Batch {batch_idx}, Img {img_idx_in_batch}\nInput Image')
    axes[0].axis('off')

    axes[1].imshow(pred_128_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Batch {batch_idx}, Img {img_idx_in_batch}\nPrediction 128x128')
    axes[1].axis('off')

    axes[2].imshow(pred_64_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Batch {batch_idx}, Img {img_idx_in_batch}\nPrediction 64x64')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # save_dir = './visualizations'
    # os.makedirs(save_dir, exist_ok=True)
    # plt.savefig(os.path.join(save_dir, f'batch_{batch_idx}_img_{img_idx_in_batch}_outputs.png'))
    # plt.close(fig)


if __name__ == "__main__":
    batch_size = 1
    num_workers = 1
    test_size = 10
    augmentation = None

    manifest_path = "../data_quick_loop/manifest.csv"
    model_path = '../pretrained_models/dr_module-1819.pth'

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        exit()
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit()

    print(f"Loading dataloaders from {manifest_path}")
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_class=PairedCTCBCTDatasetNPY,
        train_size=0,
        val_size=0,
        test_size=test_size,
        augmentation=augmentation
    )

    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Loading model from {model_path}")
    model = load_degradation_removal(save_path=model_path, trainable=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    print("\nStarting visualization of test batches...")

    visualize_batches_limit = 5
    print(f"Visualizing up to {visualize_batches_limit} batches.")

    for batch_idx, (_, batch) in enumerate(test_loader):
        if batch_idx >= visualize_batches_limit:
            print(f"Visualization limit ({visualize_batches_limit} batches) reached. Stopping.")
            break

        conditioning_batch = batch[0]

        print(f"\nVisualizing Batch {batch_idx + 1}/{visualize_batches_limit} (Batch shape: {conditioning_batch.shape})")

        for i in range(conditioning_batch.size(0)):
            single_conditioning_img = conditioning_batch[i:i+1]

            visualize_degradation_outputs(
                model=model,
                conditioning_tensor=single_conditioning_img,
                batch_idx=batch_idx,
                img_idx_in_batch=i
            )

    print("\nVisualization script finished.")