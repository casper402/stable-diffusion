import torch
import torch.nn.functional as F
from torch import optim
from models.vae import VAE 
from utils.losses import SSIMLoss, kl_divergence
from utils.train_helpers import run_training_loop
from data.dataset import get_dataloaders
from data.transforms import build_transforms
from utils.config import load_config, get_device
from functools import partial

def vae_loss_step(model, x, device, ssim_loss, beta=0.1, lambda_ssim=0.7):
    _, CT = x
    CT = CT.to(device)

    _, mu, logvar, recon = model(CT)

    recon_loss = F.l1_loss(recon, CT)
    kl = kl_divergence(mu, logvar)
    ssim = ssim_loss(recon, CT)

    total_loss = (
        recon_loss +
        beta * kl +
        lambda_ssim * ssim
        # TODO: Perceptual loss (Only if using a perceptual model trained on medical images)
    )
    return total_loss

def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False

def unfreeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = True

def main():
    device = get_device()
    config = load_config(device)
    
    train_transform, val_transform = build_transforms(config["model"]["image_size"])
    train_loader, val_loader = get_dataloaders(config, train_transform, val_transform)

    vae = VAE(latent_dim=config["model"]["latent_dim"]).to(device)

    ssim_loss = SSIMLoss()
    loss_step_fn = partial(vae_loss_step, ssim_loss=ssim_loss)

    print("\n[Phase 1] Training decoder (encoder frozen)...")
    freeze_encoder(vae)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, vae.parameters()), 
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=config["train"]["scheduler_patience"], 
        min_lr=config["train"]["min_learning_rate"]
    )    

    run_training_loop(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_step_fn=loss_step_fn,
        config=config,
        device=device,
        save_path="checkpoints/vae_decoder_only.pth",
        scheduler=scheduler
    )

    print("\n[Phase 2] Fine-tuning entire VAE...")
    unfreeze_encoder(vae)
    optimizer = optim.Adam(
        vae.parameters(), 
        lr=config["train"]["fine_tune_lr"],
        weight_decay=config["train"]["fine_tune_weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=config["train"]["scheduler_patience"], 
        min_lr=config["train"]["min_fine_tune_lr"]
    )    

    run_training_loop(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_step_fn=loss_step_fn,
        config=config,
        device=device,
        save_path="checkpoints/vae_finetuned.pth"
    )

    print("training complete")

if __name__ == "__main__":
    main()

    




