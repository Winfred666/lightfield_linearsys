#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_volume(recon_path: Path) -> np.ndarray:
    obj = torch.load(recon_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        vol = obj
    elif isinstance(obj, dict):
        if "reconstruction" not in obj:
            raise KeyError(f"Missing 'reconstruction' key in {recon_path}")
        vol = obj["reconstruction"]
    else:
        raise TypeError(f"Unsupported reconstruction type: {type(obj)}")

    if not isinstance(vol, torch.Tensor):
        raise TypeError(f"Loaded reconstruction is not a torch.Tensor: {type(vol)}")

    vol_np = vol.detach().cpu().float().numpy()
    return vol_np


def main() -> None:
    base = Path("result/solve_point")
    um_cases = [20, 60, 80, 120]
    case_dirs = [base / f"newton_crop_{um}um_1p0" / "reg0.1" for um in um_cases]
    recon_paths = [d / "reconstruction.pt" for d in case_dirs]

    for p in recon_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing reconstruction file: {p}")

    # Pass 1: global finite range for shared histogram bins
    gmin = float("inf")
    gmax = float("-inf")
    for p in recon_paths:
        arr = load_volume(p)
        finite = np.isfinite(arr)
        if not np.any(finite):
            del arr
            gc.collect()
            continue
        local_min = float(arr[finite].min())
        local_max = float(arr[finite].max())
        gmin = min(gmin, local_min)
        gmax = max(gmax, local_max)
        del arr
        gc.collect()

    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise RuntimeError("All cases contain no finite values.")
    if gmax <= gmin:
        gmax = gmin + 1e-6

    n_bins = 200
    edges = np.linspace(gmin, gmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Pass 2: histograms
    hist_rows: list[tuple[int, np.ndarray, Path]] = []
    for um, case_dir, recon_path in zip(um_cases, case_dirs, recon_paths):
        arr = load_volume(recon_path)
        vals = arr[np.isfinite(arr)]
        hist, _ = np.histogram(vals, bins=edges)

        per_case_out = case_dir / "density_histogram.png"
        per_case_out.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(centers, hist, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("Density value")
        ax.set_ylabel("Voxel count (log scale)")
        ax.set_title(f"Density Histogram ({um}um)")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(per_case_out, dpi=160)
        plt.close(fig)

        hist_rows.append((um, hist, per_case_out))

        del arr
        del vals
        gc.collect()

    # Combined 2x2 panel
    combined_out = base / "density_histogram_four_cases.png"
    combined_out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (um, hist, _) in zip(axes, hist_rows):
        ax.plot(centers, hist, linewidth=1.4)
        ax.set_yscale("log")
        ax.set_title(f"{um}um")
        ax.set_xlabel("Density value")
        ax.set_ylabel("Voxel count (log)")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(combined_out, dpi=180)
    plt.close(fig)

    print(f"global_min={gmin:.9g}")
    print(f"global_max={gmax:.9g}")
    print(f"bins={n_bins}")
    print(f"combined_png={combined_out}")
    for um, _, out in hist_rows:
        print(f"case_{um}um_png={out}")


if __name__ == "__main__":
    main()
