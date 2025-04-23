import torch
from quick_loop.vae import VAE
from scripts.evaluation import compute_mae, compute_rmse
from utils.dataset import get_dataloaders, CTDatasetWithMeta
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def postprocess(img):
    scaled = np.round(img * 1000)
    clipped = np.clip(scaled, -1000, 1000)
    return clipped

def plot_comparison(original: torch.Tensor,
                    reconstructions: list,
                    names: list = None,
                    idx: int = 0,
                    vmin: int = -1000,
                    vmax: int = 1000):
    orig = original[idx].cpu().squeeze()
    orig_np = postprocess(orig).numpy()

    # Prepare names
    n = len(reconstructions)
    names = names or [f"VAE{i+1}" for i in range(n)]

    # Compute recon numpy arrays and metrics
    recon_np_list = []
    metrics = []
    for recon in reconstructions:
        r = recon[idx].cpu().squeeze()
        r_np = postprocess(r).numpy()
        recon_np_list.append(r_np)
        mae = compute_mae(orig_np, r_np)
        rmse = compute_rmse(orig_np, r_np)
        metrics.append((mae, rmse))

    if n == 1:
        # Single VAE: show original, reconstruction, and diff
        r_np = recon_np_list[0]
        mae, rmse = metrics[0]
        diff = np.abs(orig_np - r_np)
        max_diff = diff.max()

        diff_cutoff = 100

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        # Original
        axs[0].imshow(orig_np, cmap='gray', vmin=vmin, vmax=vmax)
        axs[0].set_title('Original', fontsize=14)
        axs[0].axis('off')
        # Reconstruction
        axs[1].imshow(r_np, cmap='gray', vmin=vmin, vmax=vmax)
        axs[1].set_title(f"Reconstruction\nMAE {mae:.2f}, RMSE {rmse:.2f}", fontsize=14)
        axs[1].axis('off')
        # Diff
        im = axs[2].imshow(diff, cmap='gray', vmin=0, vmax=diff_cutoff)
        axs[2].set_title(f"Absolute Difference (0–{diff_cutoff})", fontsize=14)
        axs[2].axis('off')
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        if max_diff > diff_cutoff:
            axs[2].text(
                0.5, 0.1,
                f"⚠️ max diff = {max_diff:.2f} > {diff_cutoff}",
                ha='center', va='center',
                fontsize=12, color='red',
                transform=axs[2].transAxes
            )
        plt.tight_layout()
        plt.show()

    else:
        # Multiple VAEs: show original + each recon
        fig, axs = plt.subplots(1, n+1, figsize=((n+1)*4, 5))
        axs[0].imshow(orig_np, cmap='gray', vmin=vmin, vmax=vmax)
        axs[0].set_title('Original', fontsize=14)
        axs[0].axis('off')
        for i, (r_np, (mae, rmse), name) in enumerate(zip(recon_np_list, metrics, names)):
            ax = axs[i+1]
            ax.imshow(r_np, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}\nMAE {mae:.2f}, RMSE {rmse:.2f}", fontsize=14)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


def visualize_reconstructions(
    vaes: list,
    loader,
    device,
    names: list = None,
    num_batches: int = 1
):
    # Ensure list
    if not isinstance(vaes, (list, tuple)):
        vaes = [vaes]
    n = len(vaes)
    names = names or [f"VAE{i+1}" for i in range(n)]

    for vae in vaes:
        vae.eval()

    with torch.no_grad():
        for i, (batch, _) in enumerate(loader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            # Collect reconstructions
            recons = []
            for vae in vaes:
                _, _, _, recon = vae(batch)
                recons.append(recon)
            # Plot comparison
            plot_comparison(batch, recons, names)

def visualize_latent(vae, loader, device):
    all_mu      = []
    all_volumes = []

    print("✅ Starting encoding of all samples…")
    with torch.no_grad():
        for x, vol_idx in tqdm(loader, desc="Encoding volumes", unit="batch"):
            vols = vol_idx.cpu().tolist()
            x    = x.to(device)
            _, mu, _, _ = vae(x)
            all_mu.append(mu.cpu().numpy())
            all_volumes.extend(vols)
    print("✅ Finished encoding all samples.\n")

    all_mu = np.vstack(all_mu)
    N      = all_mu.shape[0]
    all_mu = all_mu.reshape(N, -1)
    vols   = np.array(all_volumes)

    perp = min(30, N-1)
    print(f"🌀 Running t-SNE on {N} samples (perplexity={perp})…\n")

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate='auto',
        init='random',
        verbose=2,
        random_state=42
    )
    mu_2d = tsne.fit_transform(all_mu)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        mu_2d[:,0], mu_2d[:,1],
        c=vols,
        cmap='tab20',
        s=10,
        alpha=0.8
    )
    plt.title("t-SNE of VAE latent μ (flattened; colored by volume)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    unique_vols = np.unique(vols)
    handles, _  = scatter.legend_elements(prop="colors",
                                          num=len(unique_vols))
    labels = [f"Vol {v}" for v in unique_vols]
    plt.legend(handles, labels, title="Volume",
               bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_latent_with_target(vae,
                     loader,
                     device,
                     volumes: list = None,
                     target_per_volume: int = None):
    filter_set = set(volumes) if volumes else None
    counters = {vol: 0 for vol in filter_set} if filter_set else {}
    collected_mu, collected_vols = [], []
    print("✅ Starting selective encoding of samples…")
    with torch.no_grad():
        for x, vol_idxs in tqdm(loader, desc="Encoding volumes", unit="batch"):
            for i, vol in enumerate(vol_idxs.cpu().tolist()):
                if filter_set and (vol not in filter_set or counters[vol] >= target_per_volume):
                    continue
                xi = x[i:i+1].to(device)
                _, mu, _, _ = vae(xi)
                mu_np = mu.cpu().numpy().reshape(-1)
                collected_mu.append(mu_np)
                collected_vols.append(int(vol))
                if filter_set:
                    counters[vol] += 1
                    if all(count >= target_per_volume for count in counters.values()):
                        break
            if filter_set and all(count >= target_per_volume for count in counters.values()):
                break
    print(f"✅ Collected {len(collected_vols)} samples across volumes.")
    all_mu = np.vstack(collected_mu)

    vols_arr = np.array(collected_vols)

    N = all_mu.shape[0]
    perp = min(30, N-1)

    print(f"🌀 Running t-SNE on {N} samples (perplexity={perp})…\n")
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate='auto', init='random', verbose=2, random_state=42)
    mu_2d = tsne.fit_transform(all_mu)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(mu_2d[:,0], mu_2d[:,1], c=vols_arr, cmap='tab20', s=10, alpha=0.8)

    plt.title("t-SNE of VAE latent μ (flattened; colored by volume)")
    plt.xlabel("t-SNE dim 1"); plt.ylabel("t-SNE dim 2")

    unique_vols = np.unique(vols_arr)
    handles, _ = scatter.legend_elements(prop="colors", num=len(unique_vols))
    labels = [f"Vol {v}" for v in unique_vols]
    plt.legend(handles, labels, title="Volume", bbox_to_anchor=(1.05,1), loc='upper left')

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = '../training_data/manifest-full.csv'
    loader, _, _ = get_dataloaders( # Using train loader temporarily! Change to test!
        manifest_path,
        batch_size=4,
        num_workers=4,
        dataset_class=CTDatasetWithMeta,
    )

    vae_paths = [
       "bigvae.pth", 
        # "../quickloopvae.pth"
    ]
    vaes = []
    for path in vae_paths:
        model = VAE().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        vaes.append(model)

    # visualize_reconstructions(vaes, loader, device, vae_paths, num_batches=10)
    # visualize_latent(vaes[0], loader, device)

    volumes_of_interest = [73, 2, 127, 77, 103, 120, 100, 16, 82, 29]
    target_count = 2 # should be 50
    visualize_latent_with_target(vaes[0],
                     loader,
                     device,
                     volumes=volumes_of_interest,
                     target_per_volume=target_count)

if __name__ == "__main__":
    main()
