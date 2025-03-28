import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
from torch import optim
from models.resnet50Vae import VAE 
from utils.losses import PerceptualLoss, kl_divergence
from utils.train_helpers import run_training_loop
from data.dataset import get_ct_dataloaders
from data.transforms import build_train_transform
from utils.config import load_config, get_device
from functools import partial
from piq import ssim

def vae_loss_step(model, x, device, perceptual_loss, beta=0.1, lambda_ssim=0.1, lambda_perceptual=0.1):
    CT = x.to(device)

    _, mu, logvar, recon = model(CT)

    recon_loss = F.mse_loss(recon, CT)
    kl = kl_divergence(mu, logvar)
    ssim_loss = 1 - ssim(recon.detach(), CT.detach())
    perceptual = perceptual_loss(recon.detach(), CT.detach())

    total_loss = (
        recon_loss +
        beta * kl +
        lambda_ssim * ssim_loss +
        lambda_perceptual * perceptual
    )
    return total_loss

def main():
    device = get_device()
    config = load_config(device)
    
    transform = build_train_transform(config["model"]["image_size"])
    train_loader, val_loader = get_ct_dataloaders(config, transform, subset_size=config["train"]["subset_size"])

    vae = VAE(latent_dim=config["model"]["latent_dim"]).to(device)
    perceptual_loss = PerceptualLoss(device)
    loss_step_fn = partial(vae_loss_step, perceptual_loss=perceptual_loss, beta=config["vae"]["beta"], lambda_perceptual=config["vae"]["lambda_perceptual"])

    # Phase 1: Freeze encoder
    for param in vae.encoder.parameters():
        param.requires_grad = False
    print("Phase 1: Training decoder only (encoder frozen)")

    optimizer_phase1 = optim.Adam(
        filter(lambda p: p.requires_grad, vae.parameters()), 
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler_phase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase1, 
        mode='min', 
        factor=0.5, 
        patience=config["train"]["scheduler_patience"], 
        min_lr=config["train"]["min_learning_rate"]
    )

    run_training_loop(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_phase1,
        loss_step_fn=loss_step_fn,
        epochs=config["train"]["initial_epochs"],
        config=config,
        device=device,
        save_path="checkpoints/vae_frozen_encoder.pth",
        scheduler=scheduler_phase1
    )

    # Phase 2: Unfreeze encoder for fine-tuning
    print("Phase 2: Fine-tuning entire VAE (encoder + decoder)")
    for param in vae.encoder.parameters():
        param.requires_grad = True

    optimizer_phase2 = optim.Adam(
        filter(lambda p: p.requires_grad, vae.parameters()), 
        lr=config["train"]["fine_tune_lr"], 
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler_phase2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase2, 
        mode='min', 
        factor=0.5, 
        patience=config["train"]["scheduler_patience"], 
        min_lr=config["train"]["min_learning_rate"]
    )

    run_training_loop(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_phase2,
        loss_step_fn=loss_step_fn,
        epochs=config["train"]["epochs"],
        config=config,
        device=device,
        save_path="checkpoints/vae_finetuned.pth",
        scheduler=scheduler_phase2
    )

    print("Training complete")

if __name__ == "__main__":
    main()

    




