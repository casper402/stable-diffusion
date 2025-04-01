import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
from torch import optim
from models.vae import VAE 
from utils.losses import PerceptualLoss
from data.dataset import get_ct_dataloaders
from data.transforms import build_train_transform
from utils.config import load_config, get_device
from piq import ssim
import time

def vae_loss_step(config, model, x, device, perceptual_loss, beta):
    CT = x.to(device)
    _, mu, logvar, recon = model(CT)
    l2_loss = F.mse_loss(recon, CT)
    l1_loss = F.l1_loss(recon, CT)
    kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    perceptual = perceptual_loss(recon, CT)
    ssim_loss = 1 - ssim(recon, CT)
    total_loss = (
        config["vae"]["lambda_l1"] * l1_loss +
        config["vae"]["lambda_l2"] * l2_loss +
        config["vae"]["lambda_perceptual"] * perceptual + 
        config["vae"]["lambda_ssim"] * ssim_loss +
        beta * kl_divergence
    )
    return total_loss

def validate_one_epoch(config, model, dataloader, device, perceptual_loss, beta):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = vae_loss_step(config, model, batch, device, perceptual_loss, beta)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def train_one_epoch(config, model, dataloader, optimizer, device, perceptual_loss, beta):
    model.train()
    running_loss = 0
    # TODO: simulate higher batch size with accumulation for faster training
    # TODO: Use scaler / autocast for faster training
    for batch in dataloader:
        loss = vae_loss_step(config, model, batch, device, perceptual_loss, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def run_training_loop(config, model, device, train_loader, val_loader, optimizer, perceptual_loss, scheduler):
    best_val_loss = float('inf')
    counter = 0
    beta = config["vae"]["beta"]
    epochs = config["train"]["epochs"]
    for epoch in range(epochs):
        start_time = time.time()
        if epoch > 50:
            beta = min(config["vae"]["max_beta"], beta * 1.05)
        train_loss = train_one_epoch(config, model, train_loader, optimizer, device, perceptual_loss, beta)
        val_loss = validate_one_epoch(config, model, val_loader, device, perceptual_loss, beta)
        epoch_time = time.time() - start_time
        elapsed = time.strftime("%H:%M:%S", time.gmtime(epoch_time))
        print(f"Epoch: {epoch+1}/{epochs} | Time: {elapsed} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Beta: {beta}")
        scheduler.step(val_loss) # TODO: change to val_loss
        if val_loss < best_val_loss: # TODO: change to val_loss
            best_val_loss = val_loss # TODO: change to val_loss
            counter = 0
            torch.save(model.state_dict(), config["paths"]["save_path"])
        else:
            counter += 1
            if counter >= config["train"]["early_stopping_patience"]:
                print("Early stopping")
                break

def main():
    device = get_device()
    config = load_config(device)
    transform = build_train_transform(config["model"]["image_size"])
    train_loader, val_loader = get_ct_dataloaders(config, transform)
    vae = VAE(latent_dim=config["model"]["latent_dim"]).to(device)
    perceptual_loss = PerceptualLoss(device)
    optimizer = optim.AdamW(
        vae.parameters(),
        lr=config["train"]["learning_rate"],
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
        config=config,
        model=vae,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        perceptual_loss=perceptual_loss,
        scheduler=scheduler,
    )
    print("training complete")

if __name__ == "__main__":
    main()

    




