import torch
import torch.nn.functional as F
from quick_loop.vae import VAE
from scripts.evaluation import compute_mae, compute_rmse
from utils.dataset import get_dataloaders, CTDatasetWithMeta
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.losses import PerceptualLoss, SsimLoss

# Loss weights
perceptual_weight = 0.1
ssim_weight       = 0.8
kl_weight         = 1e-5
l1_weight         = 1.0

def vae_loss(recon, x, mu, logvar):
    """
    Compute combined VAE loss.
    Returns total, perceptual, kl, ssim, l1 components.
    """
    perceptual = perceptual_loss(recon, x) * perceptual_weight
    kl = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]),
        dim=0
    ) * kl_weight
    ssim = ssim_loss(recon, x) * ssim_weight
    l1   = F.l1_loss(recon, x) * l1_weight
    total_loss = perceptual + kl + ssim + l1
    return total_loss, perceptual, kl, ssim, l1


def postprocess(img: torch.Tensor) -> np.ndarray:
    """
    Scale and clip CT Hounsfield units for visualization.
    """
    scaled = torch.round(img * 1000)
    clipped = torch.clamp(scaled, -1000, 1000)
    return clipped.cpu().numpy()

def postprocess_tanh(img: torch.Tensor, eps: float = 1e-4) -> np.ndarray:
    """
    Invert tanh-based preprocessing (tanh(HU/150)) and
    prepare for display (round + clip to [-1000, 1000]).
    """
    # 1. Clamp into (–1,1) so atanh stays finite
    x = torch.clamp(img, -1 + eps, 1 - eps)

    # 2. Inverse tanh via torch.atanh, then rescale
    hu = torch.atanh(x) * 150.0

    # 3. Round to integer HUs and clamp to [-1000,1000]
    hu = torch.round(hu)
    hu = torch.clamp(hu, -1000.0, 1000.0)

    return hu.cpu().numpy().astype(np.float32)

def add_latent_noise(z: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Add Gaussian noise to latent vector.
    """
    return z + torch.randn_like(z) * noise_std


def plot_multi_noise(
    original: torch.Tensor,
    recons_list: list,
    names: list,
    noise_levels: list,
    idx: int = 0,
    vmin: int = -1000,
    vmax: int = 1000
):
    """
    Plot multiple rows: first clean (noise=0), then for each noise level.
    recons_list: list of recon-lists, length = len(noise_levels)+1
    noise_levels: list of floats (exclude 0)
    names: list of model names
    """
    orig = original[idx].cpu().squeeze()
    orig_np = postprocess(orig)
    n_models = len(names)
    n_rows = len(recons_list)

    # Prepare metrics and losses for each recon set
    def prepare(recons, mu, logvar):
        arr, mets, los = [], [], []
        for recon, mu_i, logvar_i in zip(recons, mu, logvar):
            r = recon[idx].cpu().squeeze()
            r_np = postprocess(r)
            mae = compute_mae(orig_np, r_np)
            rmse = compute_rmse(orig_np, r_np)
            tot, perc, kl, ssim_v, l1 = vae_loss(
                recon[idx:idx+1], original[idx:idx+1], mu_i[idx:idx+1], logvar_i[idx:idx+1]
            )
            arr.append(r_np)
            mets.append((mae, rmse))
            los.append((tot.item(), perc.item(), kl.item(), ssim_v.item(), l1.item()))
        return arr, mets, los

    fig, axs = plt.subplots(n_rows, n_models+1, figsize=((n_models+1)*4, n_rows*4))

    for row, recon_set in enumerate(recons_list):
        # header title
        level = 0 if row == 0 else noise_levels[row-1]
        # first column original image constant
        ax0 = axs[row, 0] if n_rows>1 else axs[0]
        ax0.imshow(orig_np, cmap='gray', vmin=vmin, vmax=vmax)
        title = "Original" if level==0 else f"Noise σ={level:.2f}"
        ax0.set_title(title, fontsize=14)
        ax0.axis('off')

        # metrics for this set
        recon_list, metrics, losses = prepare(recon_set, mus_all[row], logvars_all[row])
        for col, (r_np, (mae, rmse), (tot, perc, kl, ssim_v, l1), name) in enumerate(
            zip(recon_list, metrics, losses, names)
        ):
            ax = axs[row, col+1] if n_rows>1 else axs[col+1]
            ax.imshow(r_np, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(
                f"{name}\nMAE {mae:.2f}, RMSE {rmse:.2f}\n"
                f"Loss {tot:.2f} (p {perc:.2f}, kl {kl:.2e}, ssim {ssim_v:.2f}, l1 {l1:.2f})",
                fontsize=10
            )
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_reconstructions(
    vaes: list,
    loader,
    device,
    names: list = None,
    num_batches: int = 1,
    noise_levels: list = None
):
    """
    Reconstruct clean and multiple latent-noisy inputs, then plot grid.
    noise_levels: list of noise stds e.g. [0.1,0.2,...]
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    if not isinstance(vaes, (list, tuple)):
        vaes = [vaes]
    names = names or [f"VAE{i+1}" for i in range(len(vaes))]
    for vae in vaes:
        vae.eval()

    with torch.no_grad():
        for i, (batch, _) in enumerate(loader):
            if i >= num_batches:
                break
            batch = batch.to(device)

            # prepare lists
            recon_sets = []
            global mus_all, logvars_all
            mus_all = []
            logvars_all = []

            # clean
            recon_clean_list, mu_list, logvar_list = [], [], []
            for vae in vaes:
                z, mu, logvar, recon_clean = vae(batch)
                recon_clean_list.append(recon_clean)
                mu_list.append(mu)
                logvar_list.append(logvar)
            recon_sets.append(recon_clean_list)
            mus_all.append(mu_list)
            logvars_all.append(logvar_list)

            # noisy levels
            for sigma in noise_levels:
                recon_noisy_list, mu_list, logvar_list = [], [], []
                for vae in vaes:
                    z, mu, logvar, _ = vae(batch)
                    z_n = add_latent_noise(z, noise_std=sigma)
                    recon_n = vae.decode(z_n)
                    recon_noisy_list.append(recon_n)
                    mu_list.append(mu)
                    logvar_list.append(logvar)
                recon_sets.append(recon_noisy_list)
                mus_all.append(mu_list)
                logvars_all.append(logvar_list)

            plot_multi_noise(
                batch,
                recon_sets,
                names,
                noise_levels
            )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global perceptual_loss, ssim_loss
    perceptual_loss = PerceptualLoss(device=device)
    ssim_loss       = SsimLoss()

    manifest_path = '../training_data/manifest-full.csv'
    _, _, loader = get_dataloaders(
        manifest_path,
        batch_size=4,
        num_workers=4,
        dataset_class=CTDatasetWithMeta,
    )

    vae_paths = [
        "vaeV6.pth",
        "casper_vae.pth",
        "vae_joint_v2.pth",
    ]
    vaes = []
    for path in vae_paths:
        model = VAE().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        vaes.append(model)

    visualize_reconstructions(
        vaes, loader, device,
        names=vae_paths,
        num_batches=10,
        noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]
    )

if __name__ == "__main__":
    main()
