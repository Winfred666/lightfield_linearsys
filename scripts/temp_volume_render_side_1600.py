#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_volume(recon_path: Path) -> torch.Tensor:
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

    return vol.detach().cpu().float()


def compute_stats(vol: torch.Tensor) -> dict[str, float]:
    finite_mask = torch.isfinite(vol)
    if bool(torch.all(finite_mask)):
        values = vol.reshape(-1)
    else:
        values = vol[finite_mask]

    if values.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "median": float("nan"), "mean": float("nan")}

    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "median": float(values.median().item()),
        "mean": float(values.mean().item()),
    }


def save_side_render(vol: torch.Tensor, out_path: Path, vmax: float = 1600.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proj = np.nansum(vol.numpy(), axis=0)  # (Y, Z)

    h, w = proj.shape
    w_in = 6.0
    h_in = max(3.0, w_in * (h / max(w, 1)))
    fig, ax = plt.subplots(figsize=(w_in, h_in))

    if proj.size > 0:
        vmin = float(np.nanpercentile(proj, 0))
    else:
        vmin = 0.0

    im = ax.imshow(
        proj,
        cmap="viridis",
        aspect="equal",
        origin="lower",
        vmin=vmin,
        vmax=float(vmax),
    )
    ax.set_title("Side View (Sum projection along X)\nLeft->Right is Z-axis")
    ax.set_xlabel("Z (Depth)")
    ax.set_ylabel("Y (Vertical)")
    fig.colorbar(im, ax=ax, label="Projected density (sum over X)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    base = Path("result/solve_point")
    um_cases = [20, 60, 80, 120]

    rows: list[tuple[int, dict[str, float], Path]] = []

    for um in um_cases:
        case_dir = base / f"newton_crop_{um}um_1p0" / "reg0.1"
        recon_path = case_dir / "reconstruction.pt"
        out_png = case_dir / "volume_render_side_1600.png"

        if not recon_path.exists():
            raise FileNotFoundError(f"Missing reconstruction file: {recon_path}")

        vol = load_volume(recon_path)
        stats = compute_stats(vol)
        save_side_render(vol, out_png, vmax=1600.0)

        rows.append((um, stats, out_png))

        del vol
        gc.collect()

    print("case_um,min,max,median,mean,output_png")
    for um, stats, out_png in rows:
        print(
            f"{um},"
            f"{stats['min']:.9g},"
            f"{stats['max']:.9g},"
            f"{stats['median']:.9g},"
            f"{stats['mean']:.9g},"
            f"{out_png}"
        )


if __name__ == "__main__":
    main()
