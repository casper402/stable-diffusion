#!/usr/bin/env python
import os
import numpy as np
import nibabel as nib
import concurrent.futures

# Optional dependencies for advanced modules
try:
    from sklearn.linear_model import RANSACRegressor, LinearRegression
except ImportError:
    RANSACRegressor = None
try:
    from skimage.exposure import match_histograms
except ImportError:
    match_histograms = None
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# ------------------------
# Module: Loading & Mask
# ------------------------
def load_volumes(ct_path, cbct_path):
    """Load CT and CBCT volumes (returns two 3D numpy arrays)."""
    ct = nib.load(ct_path).get_fdata()
    cbct = nib.load(cbct_path).get_fdata()
    return ct, cbct


def mask_body(ct_vol, lower=-900, upper=3000):
    """Create a boolean mask of voxels within [lower, upper] HU."""
    return (ct_vol > lower) & (ct_vol < upper)

# ---------------------------------
# Module: Sampling & Subsampling
# ---------------------------------
def sample_random(ct_vol, cbct_vol, mask=None, n_samples=20000):
    """
    Randomly sample up to n_samples valid CT/CBCT voxel pairs.
    If mask is given, only sample within the mask.
    Returns arrays x (CBCT) and y (CT).
    """
    flat_ct   = ct_vol.ravel()
    flat_cbct = cbct_vol.ravel()
    if mask is not None:
        flat_mask = mask.ravel()
        valid = flat_mask & np.isfinite(flat_ct) & np.isfinite(flat_cbct)
    else:
        valid = np.isfinite(flat_ct) & np.isfinite(flat_cbct)
    idxs = np.where(valid)[0]
    n = min(n_samples, idxs.size)
    chosen = np.random.choice(idxs, size=n, replace=False)
    return flat_cbct[chosen], flat_ct[chosen]


def sample_stratified(ct_vol, cbct_vol, mask=None, bins=50, per_bin=400):
    """
    Stratified sampling: split CT range into bins and sample equally from each.
    """
    flat_ct   = ct_vol.ravel()
    flat_cbct = cbct_vol.ravel()
    valid = np.isfinite(flat_ct) & np.isfinite(flat_cbct)
    if mask is not None:
        valid &= mask.ravel()
    idxs = np.where(valid)[0]
    hu_vals = flat_ct[idxs]
    bin_edges = np.linspace(hu_vals.min(), hu_vals.max(), bins+1)
    bin_idx = np.digitize(hu_vals, bin_edges)
    sampled = []
    for b in range(1, bins+1):
        bin_members = idxs[bin_idx == b]
        if bin_members.size:
            count = min(per_bin, bin_members.size)
            sampled.append(
                np.random.choice(bin_members, size=count, replace=False)
            )
    all_idx = np.concatenate(sampled)
    return flat_cbct[all_idx], flat_ct[all_idx]

# ------------------------
# Module: Fitting Models
# ------------------------
def fit_linear(x, y):
    """Ordinary least squares: y ≈ a*x + b"""
    a, b = np.polyfit(x, y, 1)
    return a, b


def fit_ransac(x, y, residual_threshold=50, max_trials=100):
    """RANSAC-based robust linear fit y ≈ a*x + b. Requires sklearn."""
    if RANSACRegressor is None:
        raise ImportError("scikit-learn is required for RANSAC fitting")
    model = RANSACRegressor(base_estimator=LinearRegression(),
                            residual_threshold=residual_threshold,
                            max_trials=max_trials)
    model.fit(x.reshape(-1,1), y)
    a = model.estimator_.coef_[0]
    b = model.estimator_.intercept_
    return a, b


def fit_polynomial(x, y, degree=2):
    """Polynomial fit y ≈ c0*x^2 + c1*x + c2 (degree=2)."""
    coeffs = np.polyfit(x, y, degree)
    return coeffs  # highest-to-lowest power

# --------------------------------
# Module: Pre/Post Processing
# --------------------------------
def histogram_match_volumes(cbct_vol, ct_vol):
    """Histogram match CBCT to CT. Requires scikit-image."""
    if match_histograms is None:
        raise ImportError("scikit-image is required for histogram_matching")
    return match_histograms(cbct_vol, ct_vol, multichannel=False)


def bias_field_correction(cbct_vol):
    """Apply N4 bias-field correction. Requires SimpleITK."""
    if sitk is None:
        raise ImportError("SimpleITK is required for bias-field correction")
    img = sitk.GetImageFromArray(cbct_vol)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    out = corrector.Execute(img)
    return sitk.GetArrayFromImage(out)

# ------------------------
# Module: Apply & Save
# ------------------------
def apply_calibration(cbct_vol, scale, offset, clip_range=(-1000, 1000)):
    """Apply linear mapping and clip"""
    vol = cbct_vol * scale + offset
    return np.clip(vol, clip_range[0], clip_range[1])


def apply_polynomial(cbct_vol, coeffs, clip_range=(-1000,1000)):
    """Apply polynomial mapping and clip"""
    # coeffs[0]*x^2 + coeffs[1]*x + coeffs[2]
    vol = np.polyval(coeffs, cbct_vol)
    return np.clip(vol, clip_range[0], clip_range[1])


def rotate_and_save_slices(vol, base, output_dir, debug=False):
    for i in range(vol.shape[2]):
        slice_ = vol[:, :, i]
        rot = np.rot90(slice_, k=-1)
        fname = f"{base}_slice_{i:03d}.npy"
        np.save(os.path.join(output_dir, fname), rot)
        if debug:
            print(f" Saved {fname}")

# ------------------------
# Pipeline Integration
# ------------------------
def process_nifti_pair(ct_folder, ct_fname, cbct_folder, cbct_fname,
                       output_dir, config, debug=False):
    # 1) Load
    ct_path   = os.path.join(ct_folder,   ct_fname)
    cbct_path = os.path.join(cbct_folder, cbct_fname)
    ct_vol, cbct_vol = load_volumes(ct_path, cbct_path)

    # 2) Optional bias correction
    if config.get("bias_correction", False):
        cbct_vol = bias_field_correction(cbct_vol)

    # 3) Optional histogram matching
    if config.get("histogram_match", False):
        cbct_vol = histogram_match_volumes(cbct_vol, ct_vol)

    # 4) Mask
    mask = None
    if config.get("mask_body", False):
        mask = mask_body(ct_vol,
                         config.get("mask_lower", -900),
                         config.get("mask_upper", 3000))

    # 5) Sampling
    samp_cfg = config.get("sampling", {})
    method   = samp_cfg.get("method", "random")
    if method == "random":
        x, y = sample_random(ct_vol, cbct_vol, mask,
                             samp_cfg.get("n_samples", 20000))
    else:
        x, y = sample_stratified(ct_vol, cbct_vol, mask,
                                 samp_cfg.get("bins", 50),
                                 samp_cfg.get("per_bin", 400))

    # 6) Fit
    fit_cfg = config.get("fitting", {})
    fmethod = fit_cfg.get("method", "linear")
    if fmethod == "ransac":
        scale, offset = fit_ransac(x, y,
                                   fit_cfg.get("residual_threshold", 50),
                                   fit_cfg.get("max_trials", 100))
        apply_fn = apply_calibration
    elif fmethod == "poly2":
        coeffs = fit_polynomial(x, y, degree=2)
        apply_fn = lambda vol: apply_polynomial(vol, coeffs)
    else:
        scale, offset = fit_linear(x, y)
        apply_fn = apply_calibration

    if debug and fmethod != "poly2":
        print(f"Fitted {fmethod}: scale={scale:.6f}, offset={offset:.1f}")

    # 7) Apply mapping & clip
    cbct_calib = apply_fn(cbct_vol) if fmethod == "poly2" else apply_fn(cbct_vol, scale, offset)

    # 8) Save
    base = os.path.splitext(cbct_fname)[0]
    if base.startswith("REC-"):
        base = "volume-" + base[4:]
    rotate_and_save_slices(cbct_calib, base, output_dir, debug)

    if debug:
        print(f"[Volume {ct_fname}] Done; saved {cbct_calib.shape[2]} slices in {output_dir}")

# ------------------------
# Main: multithreaded batch
# ------------------------
def main():
    # Config: toggle modules here
    config = {
        "bias_correction": False,
        "histogram_match": False,
        "mask_body":      True,
        "mask_lower":     -700,
        "mask_upper":     1000,
        "sampling": {
            "method":    "random",  # or "stratified"
            "n_samples": 20000,
            "bins":      50, # Only relevant for stratified
            "per_bin":   400, # Only relevant for stratified
        },
        "fitting": {
            "method":             "linear",  # "linear", "ransac", or "poly2"
            "residual_threshold": 50,
            "max_trials":         100,
        }
    }

    ct_folder   = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCTAlignedToCBCT"
    cbct_folder = "/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated/256"
    output_dir  = "/Users/Niklas/thesis/training_data/CBCT/scaledV3"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    num_volumes = 1 # 131
    max_workers = min(8, (os.cpu_count() or 1))

    def worker(idx):
        ct_fname   = f"volume-{idx}.nii"
        cbct_fname = f"REC-{idx}.nii"
        process_nifti_pair(ct_folder, ct_fname,
                           cbct_folder, cbct_fname,
                           output_dir, config, debug=False)
        return idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(worker, i): i for i in range(num_volumes)}
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                fut.result()
                print(f"[Main] Volume {i} done.")
            except Exception as e:
                print(f"[Main] Volume {i} error: {e}")

if __name__ == "__main__":
    main()
