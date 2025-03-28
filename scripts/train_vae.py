import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
from torch import optim
from models.vae import VAE 
from utils.losses import PerceptualLoss, kl_divergence
from utils.train_helpers import run_training_loop
from data.dataset import get_ct_dataloaders
from data.transforms import build_train_transform
from utils.config import load_config, get_device
from functools import partial

def vae_loss_step(model, x, device, perceptual_loss, beta=0.00001, lambda_perceptual=0.0001):
    CT = x.to(device)

    _, mu, logvar, recon = model(CT)

    recon_loss = F.mse_loss(recon, CT, reduction='mean')
    kl = kl_divergence(mu, logvar)
    perceptual = perceptual_loss(recon.detach(), CT.detach())

    total_loss = (
        recon_loss +
        beta * kl +
        lambda_perceptual * perceptual
    )
    return total_loss

def main():
    device = get_device()
    config = load_config(device)
    
    transform = build_train_transform(config["model"]["image_size"])
    train_loader, val_loader = get_ct_dataloaders(config, transform, subset_size=config["train"]["subset_size"])

    vae = VAE(latent_dim=4).to(device)
    perceptual_loss = PerceptualLoss(device)
    loss_step_fn = partial(vae_loss_step, perceptual_loss=perceptual_loss)

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=2e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',            # minimize reconstruction or total loss
        factor=0.5,            # LR reduction factor (usually 0.5–0.1)
        patience=10,           # epochs without improvement before reduction
        threshold=1e-4,        # significant improvement threshold
        verbose=True,          # prints LR updates to keep track
        min_lr=1e-6            # minimal LR allowed
    ) 

    run_training_loop(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_step_fn=loss_step_fn,
        epochs=config["train"]["epochs"],
        config=config,
        device=device,
        save_path="checkpoints/test2.pth",
        scheduler=scheduler
    )

    print("training complete")

if __name__ == "__main__":
    main()

    




