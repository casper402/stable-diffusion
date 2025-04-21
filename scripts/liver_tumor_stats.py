#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures as cf
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ─────────────────────────────────── Low‑level helpers ──────────────────────────────────

def compute_masked_stats(image: np.ndarray, mask: np.ndarray) -> tuple[int, np.float64, np.float64]:
    """Return *count*, *sum* and *sum‑of‑squares* of pixel values under *mask*."""
    m = mask if mask.dtype == bool else mask.astype(bool, copy=False)
    cnt: int = int(m.sum())
    if cnt == 0:
        return 0, np.float64(0), np.float64(0)

    vals = image[m].astype(np.float64, copy=False)
    return cnt, vals.sum(dtype=np.float64), np.square(vals, dtype=np.float64).sum(dtype=np.float64)


def variance(cnt: int, s: float, ssq: float) -> float:
    """Population variance from aggregated terms (two‑pass, numerically stable)."""
    return (ssq - (s * s) / cnt) / cnt if cnt else 0.0

# ────────────────────────────────── Per‑volume processing ──────────────────────────────

def compare_volume(
    vol_idx: int,
    img_dir: Path,
    liver_dir: Path,
    tumour_dir: Path,
) -> tuple[float, float, float, float, int, float, float, int, float, float] | None:
    """Compute stats for one volume, returning a packed tuple (see below)."""
    files = sorted(img_dir.glob(f"volume-{vol_idx}_slice_*.npy"))
    if not files:
        logger.warning("No image slices found for volume %d", vol_idx)
        return None

    l_slice_means: list[float] = []
    t_slice_means: list[float] = []

    l_cnt = l_sum = l_ssq = 0
    t_cnt = t_sum = t_ssq = 0

    for fp in files:
        try:
            img = np.load(fp)
            lm = np.load(liver_dir / fp.name)
            tm = np.load(tumour_dir / fp.name)
        except FileNotFoundError as e:
            logger.error("Missing file for volume %d: %s", vol_idx, e)
            continue

        # — liver —
        lc, ls, lss = compute_masked_stats(img, lm)
        if lc:
            l_slice_means.append(ls / lc)
        l_cnt += lc; l_sum += ls; l_ssq += lss

        # — tumour —
        tc, ts, tss = compute_masked_stats(img, tm)
        if tc:
            t_slice_means.append(ts / tc)
        t_cnt += tc; t_sum += ts; t_ssq += tss

    # Per‑volume μ/σ of slice means (sample σ, ddof=1)
    vol_l_mean = float(np.mean(l_slice_means)) if l_slice_means else np.nan
    vol_l_sd   = float(np.std(l_slice_means, ddof=1)) if l_slice_means else np.nan
    vol_t_mean = float(np.mean(t_slice_means)) if t_slice_means else np.nan
    vol_t_sd   = float(np.std(t_slice_means, ddof=1)) if t_slice_means else np.nan

    return (
        vol_l_mean, vol_l_sd,
        vol_t_mean, vol_t_sd,
        l_cnt, l_sum, l_ssq,
        t_cnt, t_sum, t_ssq,
    )

# ───────────────────────────────── Aggregation & reports ───────────────────────────────

def aggregate_and_report(results: dict[int, tuple], *, ddof_global: int = 0) -> None:
    """Print human‑readable summary tables from the collected per‑volume tuples."""
    print("\nPer‑volume slice‑mean statistics (μ±σ)")
    for v in sorted(results):
        vlm, vls, vtm, vts, *_ = results[v]
        print(f"Volume {v:3d} │ liver {vlm:7.1f} ± {vls:5.1f} │ tumour {vtm:7.1f} ± {vts:5.1f}")

    # Mean of per‑volume slice means
    arr = np.array([results[v][:4] for v in sorted(results)])
    byvol_l_mean = np.nanmean(arr[:, 0])
    byvol_l_sd   = np.nanstd(arr[:, 1], ddof=ddof_global)
    byvol_t_mean = np.nanmean(arr[:, 2])
    byvol_t_sd   = np.nanstd(arr[:, 3], ddof=ddof_global)

    print("\nMean of per‑volume slice means (μ±σ)")
    print(f"Liver  : {byvol_l_mean:7.1f} ± {byvol_l_sd:5.1f}")
    print(f"Tumour : {byvol_t_mean:7.1f} ± {byvol_t_sd:5.1f}")

    # Pixel‑weighted global stats
    total = {"liver": {"cnt": 0, "sum": 0.0, "ssq": 0.0},
             "tumour": {"cnt": 0, "sum": 0.0, "ssq": 0.0}}
    for res in results.values():
        *_slice, lc, ls, lss, tc, ts, tss = res
        total["liver"]["cnt"]  += lc
        total["liver"]["sum"]  += ls
        total["liver"]["ssq"]  += lss
        total["tumour"]["cnt"] += tc
        total["tumour"]["sum"] += ts
        total["tumour"]["ssq"] += tss

    print("\nPixel‑weighted global statistics (population)")
    for region, info in total.items():
        cnt, s, ssq = info["cnt"], info["sum"], info["ssq"]
        if cnt:
            mu = s / cnt
            sigma = np.sqrt(variance(cnt, s, ssq))
            print(f"{region.title():6} │ pixels={cnt:>9d} μ={mu:7.1f} σ={sigma:5.1f}")
        else:
            print(f"{region.title():6} │ no pixels found")

# ─────────────────────────────────── Pipeline runner ───────────────────────────────────

def run_pipeline(vol_indices: list[int], img_dir: Path, liver_dir: Path, tumour_dir: Path, *, workers: int | None = None) -> dict[int, tuple]:
    """Parallel per‑volume execution with a tqdm progress bar."""
    results: dict[int, tuple] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(compare_volume, v, img_dir, liver_dir, tumour_dir): v for v in vol_indices}
        with tqdm(total=len(futures), desc="Volumes", unit="vol") as pb:
            for fut in cf.as_completed(futures):
                v = futures[fut]
                try:
                    res = fut.result()
                    if res is not None:
                        results[v] = res
                except Exception:
                    logger.exception("Volume %d failed", v)
                finally:
                    pb.update(1)
    return results

# ────────────────────────────────────────── Main ───────────────────────────────────────
if __name__ == "__main__":
    # ─── CONFIG ────────────────────────────────────────────────────────────
    img_dir   = Path("/Users/Niklas/thesis/training_data/CBCT/scaledV2").expanduser()
    liver_dir = Path("/Users/Niklas/thesis/training_data/masks/liver").expanduser()
    tumour_dir = Path("/Users/Niklas/thesis/training_data/masks/tumor").expanduser()

    volumes = [
        68, 27, 52, 104, 130, 16, 24, 75, 124, 26, 64, 90, 50, 86, 122,
        106, 65, 62, 128, 69, 15, 117, 96, 3, 76, 109, 18, 120, 73, 79,
        83, 14, 58, 17, 112, 13, 110, 125, 1, 126, 93, 51, 107, 91, 85,
        82, 67, 102, 94, 56, 84, 53, 100, 11, 48, 101, 57, 55, 80, 39,
        5, 49, 78, 129, 123, 7, 10, 88, 121, 95, 127, 92, 105, 116, 6,
        19, 115, 97, 2, 118, 66, 54, 25, 63, 108, 22, 113, 8, 111, 114,
        9, 74, 21, 77, 20, 103, 70, 87, 119, 4,
    ]

    # Number of threads (None → os.cpu_count())
    workers = None

    logging.basicConfig(level="INFO", format="%(levelname)s %(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    # ───────────────────────────────────────────────────────────────────────

    res = run_pipeline(volumes, img_dir, liver_dir, tumour_dir, workers=workers)
    if res:
        aggregate_and_report(res)
    else:
        logger.error("No volumes processed successfully")
        sys.exit(1)
