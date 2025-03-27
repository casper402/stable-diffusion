import torch
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(model, dataloader, loss_step_fn, optimizer, device, scaler):
    model.train()
    running_loss = 0

    # TODO: simulate batch size 8 with accumulation

    for batch in dataloader:
        optimizer.zero_grad()

        with autocast():
            loss = loss_step_fn(model, batch, device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_step_fn, device):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            loss = loss_step_fn(model, batch, device)
            running_loss += loss.item()

    return running_loss / len(dataloader)

def run_training_loop(model, train_loader, val_loader, optimizer, loss_step_fn, epochs, config, device, save_path, scheduler=None):
    best_val_loss = float('inf')
    counter = 0
    scaler = GradScaler()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_step_fn, optimizer, device, scaler)
        val_loss = validate_one_epoch(model, val_loader, loss_step_fn, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{config['train']['epochs']}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= config["train"]["early_stopping_patience"]:
                print("Early stopping")
                break
