import os
import argparse
import nibabel as nib
import numpy as np
import multiprocessing as mp
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from quick_loop.vae import load_vae
from quick_loop.unetControlPACA import load_unet_control_paca
from quick_loop.controlnet import load_controlnet
from quick_loop.degradationRemoval import load_degradation_removal
from models.diffusion import Diffusion

# --- Configs ---
mode = 'construct'  # options: 'construct' | 'vis'
volume_idx = '0'
ct_path = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT/volume-' + volume_idx + '.nii' # path to CT volume (or None)

# - Constructing sCT -
cbct_path = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated/256/REC-' + volume_idx + '.nii' # path to CBCT volume
align_to_CT = True
diff_timesteps = 2 # TODO: Should be 1000, but temporarily using 2, just for quick feedback loop on dimensions
interpolation_method = 'NN'  # options: 'NN' | 'bspline'
save_npy_slices = True
output_dir = './pipeline-output'

# - Visualizing -
include_absolute_difference = True

# - Stable diffusion -
sd_dir = '/Volumes/Lenovo PS8/quick_loop_results/'
vae_path = os.path.join(sd_dir, 'vae.pth')
unet_path = os.path.join(sd_dir, 'unet.pth')
controlnet_path = os.path.join(sd_dir, 'controlnet.pth')
paca_path = os.path.join(sd_dir, 'paca_layers.pth')
dr_path = os.path.join(sd_dir, 'dr_module.pth')
# ---------------


# --- Multiprocess helpers ---
mp.set_start_method('spawn', force=True)
sd_process = None
def init_worker():
    global sd_process
    sd_process = get_sd_pipeline()


def worker(slice_arr):
    return sd_process(torch.from_numpy(slice_arr).float())

# ---------------------------

def load_volume(path):
    """
    Load a NIfTI volume and return the data array and affine.
    """
    img = nib.load(path)
    return img.get_fdata(), img.affine


def align_CBCT_to_CT_range(cbct, ct):
    """
    Estimate linear mapping to align CBCT intensities to CT Hounsfield units.
    """
    x = cbct.ravel()
    y = ct.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    idxs = np.where(mask)[0]
    n = min(20000, idxs.size)
    sample = np.random.choice(idxs, n, replace=False)
    a, b = np.polyfit(x[sample], y[sample], 1)
    return np.clip(cbct * a + b, -1000, 1000)


def clip(volume):
    """
    Clip volume intensities to the range [-1000, 1000].
    """
    return np.clip(volume, -1000, 1000)


def slice_volume(volume):
    """
    Split a 3D volume into a list of 2D slices along the z-axis.
    """
    return [volume[:, :, i] for i in range(volume.shape[2])]


def get_sd_pipeline(guidance_scale=1.0):
    """
    Initialize and return a function that processes a single CBCT slice through the
    Stable-Diffusion + PACA pipeline, outputting a numpy array in [-1000, 1000] HU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    vae = load_vae(vae_path).to(device).eval()
    unet = load_unet_control_paca(
        unet_save_path=unet_path,
        paca_save_path=paca_path
    ).to(device).eval()
    controlnet = load_controlnet(save_path=controlnet_path).to(device).eval()
    dr_module = load_degradation_removal(save_path=dr_path).to(device).eval()
    diffusion = Diffusion(device, timesteps = diff_timesteps)

    def sd_process(cbct_slice_tensor: torch.Tensor) -> np.ndarray:
        if cbct_slice_tensor.dim() != 2:
            raise ValueError(
                f"Input tensor must be 2D, got {cbct_slice_tensor.dim()}D"
            )

        x = cbct_slice_tensor.to(device).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            control_input, _ = dr_module(x)
            z_t = torch.randn_like(vae.encode(x)[0])
            T = diffusion.timesteps

            for t_int in range(T-1, -1, -1):
                t = torch.full(
                    (z_t.size(0),), t_int,
                    device=device, dtype=torch.long
                )
                down, mid = controlnet(z_t, control_input, t)
                cond = unet(z_t, t, down, mid)
                uncond = unet(z_t, t, None, None)
                noise_pred = uncond + guidance_scale * (cond - uncond)

                beta = diffusion.beta[t_int].view(-1,1,1,1)
                alpha = diffusion.alpha[t_int].view(-1,1,1,1)
                acp = diffusion.alpha_cumprod[t_int].view(-1,1,1,1)
                mean_coef = beta / torch.sqrt(1 - acp)
                model_mean = torch.sqrt(1/alpha) * (
                    z_t - mean_coef * noise_pred
                )

                if t_int > 0:
                    z_t = (
                        model_mean +
                        torch.sqrt(beta) * torch.randn_like(z_t)
                    )
                else:
                    z_t = model_mean

            generated = vae.decode(z_t)
            out = (generated / 2 + 0.5).clamp(0,1)
            hu = out * 2000 - 1000

            return hu.squeeze(0).cpu().numpy()

    return sd_process


def interpolate(slices, method='NN'):
    """
    Stack and interpolate 2D slices back into a 3D volume.
    """
    vol = np.stack(slices, axis=2)

    if method == 'NN':
        return vol
    elif method == 'bspline':
        return zoom(vol, (1,1,1), order=3)
    else:
        raise ValueError(f"Unknown interpolation: {method}")


def plot_3d(volume,
            title='Volume',
            threshold=0,
            step_size=1,
            alpha=0.3):
    """
    Render a 3D mesh of the volume using Marching Cubes.
    """
    print("[plot_3d] marching cubes")
    verts, faces, _, _ = measure.marching_cubes(
        volume,
        level=threshold,
        step_size=step_size
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    print("[plot_3d] Poly3dCollection")
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor([0.5, 0.5, 0.5])

    ax.add_collection3d(mesh)
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    ax.set_title(title)

    plt.tight_layout()
    print("[plot_3d] show")
    plt.show()


def construct():
    """
    Full pipeline: CBCT → slice → SD → stitch → save sCT.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[construct] Loading CBCT from {cbct_path}")
    cbct_vol, cbct_affine = load_volume(cbct_path)

    if align_to_CT and ct_path:
        print(f"[construct] Aligning to CT from {ct_path}")
        ct_vol, _ = load_volume(ct_path)
        cbct_vol = align_CBCT_to_CT_range(cbct_vol, ct_vol)

    cbct_vol = clip(cbct_vol)
    slices = slice_volume(cbct_vol)

    print("[construct] Running inference on slices")
    with mp.Pool(mp.cpu_count(), initializer=init_worker) as pool:
        out_slices = list(
            tqdm(
                pool.imap(worker, slices),
                total=len(slices),
                desc="Slices"
            )
        )

        if save_npy_slices:
            for idx, out in enumerate(out_slices):
                fname = f"slice_{idx:03d}.npy"
                np.save(os.path.join(output_dir, fname), out)

    print("[construct] Stitching volume and saving")
    sct_vol = interpolate(out_slices, method=interpolation_method)

    print("[construct] Saving as .nii")
    out_nifti = nib.Nifti1Image(sct_vol.astype(np.float32), cbct_affine)
    nib.save(out_nifti, os.path.join(output_dir, "sCT_volume.nii"))


def visualize():
    """
    3D rendering of sCT (and CT / diff if available).
    """
    print("[visualize] Loading sCT volume")
    sct_vol, _ = load_volume(os.path.join(output_dir, "sCT_volume.nii"))
    sct_vol = clip(sct_vol)

    if ct_path:
        print(f"[visualize] Loading CT from {ct_path}")
        ct_vol, _ = load_volume(ct_path)
        ct_vol = clip(ct_vol)

        if include_absolute_difference:
            diff_vol = np.abs(ct_vol - sct_vol)

        plot_3d(sct_vol, title='sCT Volume')
        plot_3d(ct_vol, title='CT Volume')

        if include_absolute_difference:
            plot_3d(diff_vol, title='Absolute Difference')
    else:
        plot_3d(sct_vol, title='sCT Volume')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBCT-to-sCT pipeline.")
    parser.add_argument(
        '--mode', choices=['construct', 'vis'], default=mode
    )
    parser.add_argument('--ct', type=str, default=ct_path)
    parser.add_argument('--cbct', type=str, default=cbct_path)
    parser.add_argument('--out', type=str, default=output_dir)
    args = parser.parse_args()

    mode = args.mode
    ct_path = args.ct
    cbct_path = args.cbct
    output_dir = args.out

    if mode == 'construct':
        construct()
    elif mode == 'vis':
        visualize()
    else:
        raise ValueError(f"Unknown mode: {mode}")
