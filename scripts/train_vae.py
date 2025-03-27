import torch
import torch.nn.functional as F
from torch import optim
from models.vae import VAE 
from utils.losses import SSIMLoss, PerceptualLoss, kl_divergence
from utils.train_helpers import run_training_loop
from data.dataset import get_dataloaders
from data.transforms import build_train_transform
from utils.config import load_config, get_device
from functools import partial
from piq import ssim

def vae_loss_step(model, x, device, perceptual_loss, beta=0.1, lambda_ssim=0.2, lambda_perceptual=0.2):
    _, CT = x
    CT = CT.to(device)

    _, mu, logvar, recon = model(CT)

    recon_loss = F.mse_loss(recon, CT)
    kl = kl_divergence(mu, logvar)
    ssim_loss = 1 - ssim(recon, CT)

    total_loss = (
        recon_loss +
        beta * kl +
        lambda_ssim * ssim_loss +
        lambda_perceptual * perceptual_loss
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
    
    transform = build_train_transform(config["model"]["image_size"])
    train_loader, val_loader = get_dataloaders(config, transform)

    vae = VAE(latent_dim=config["model"]["latent_dim"]).to(device)
    perceptual_loss = PerceptualLoss()

    loss_step_fn = partial(vae_loss_step, perceptual_loss=perceptual_loss)

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
        save_path="checkpoints/vae_ct_only.pth",
        scheduler=scheduler
    )

    print("training complete")

if __name__ == "__main__":
    main()

    




