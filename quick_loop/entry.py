from quick_loop.vae import load_vae, train_vae

vae = load_vae()
# TODO: load CT only dataset for vae train. 
# TODO: Maybe make some function to get loaders from in utils/dataset.py
train_vae(train_loader, val_loader, early_stopping=50)
