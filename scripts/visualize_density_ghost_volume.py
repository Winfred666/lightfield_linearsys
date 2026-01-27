#!/usr/bin/env python3

"""Visualize reconstruction + ghost volumes as 2D slice videos.

This script is a non-interactive companion to `visualize_light_field.py`.
It loads a `reconstruction.pt` (or a directory containing it) and, if present,
unions all `ghost_volume/*.pt` under the same run folder.

It then renders 2D slices and saves Z-scan videos under `<run_dir>/viz/`.

Outputs (always generated):
- density_xy_no_ghost.mp4
- density_xy_ghost_full.mp4
- density_xy_ghost_thr025.mp4
- density_xy_ghost_thr050.mp4

Notes
-----
- Density is rendered with the `viridis` colormap.
- The density colorbar is consistent across all output videos.
- Ghost overlay is rendered as a magenta RGBA layer on top of the density.
  For thresholded modes, the per-frame ghost is thresholded relative to the
  *global max* of the union ghost volume.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Matplotlib is used for deterministic, headless rendering.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402


def _resolve_reconstruction_path(p: Path) -> Path:
    """Accept either a reconstruction.pt file or a directory containing one."""
    if p.is_dir():
        cand = p / "reconstruction.pt"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"No reconstruction.pt found in directory: {p}")
    return p


def _load_reconstruction_volume(pt_path: Path) -> np.ndarray:
    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, dict) and "reconstruction" in data:
        vol = data["reconstruction"]
    else:
        # Fallback: some pipelines may save the tensor directly.
        vol = data
    return vol.detach().cpu().float().numpy()


def _load_ghost_union(input_dir: Path, target_shape: tuple[int, int, int]) -> Optional[np.ndarray]:
    """Union ghost volumes under input_dir/ghost_volume/*.pt into a single float32 volume."""
    ghost_dir = input_dir / "ghost_volume"
    if not ghost_dir.exists() or not ghost_dir.is_dir():
        return None

    pt_files = sorted([p for p in ghost_dir.glob("*.pt") if p.is_file()])
    if not pt_files:
        return None

    union = np.zeros(target_shape, dtype=np.float32)
    for p in pt_files:
        d = torch.load(p, map_location="cpu")
        if isinstance(d, dict):
            if "ghost" in d:
                t = d["ghost"]
            elif "ghost_volume" in d:
                t = d["ghost_volume"]
            else:
                tensor_like = None
                for v in d.values():
                    if torch.is_tensor(v):
                        tensor_like = v
                        break
                if tensor_like is None:
                    continue
                t = tensor_like
        else:
            t = d

        arr = t.detach().cpu().float().numpy()
        if arr.shape != target_shape:
            raise ValueError(
                f"Ghost volume shape mismatch for {p}: got {arr.shape}, expected {target_shape}"
            )
        union += arr

    return union


def _robust_vmax(vol: np.ndarray, p: float = 99.0) -> float:
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return 1e-12
    v = float(np.percentile(finite, p))
    return max(v, 1e-12)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a square structuring element (no scipy dependency)."""
    if radius <= 0:
        return mask
    m = mask.astype(bool, copy=False)
    out = m.copy()
    for _ in range(radius):
        # 8-neighborhood dilation implemented via shifts.
        nbr = out.copy()
        nbr[:-1, :] |= out[1:, :]
        nbr[1:, :] |= out[:-1, :]
        nbr[:, :-1] |= out[:, 1:]
        nbr[:, 1:] |= out[:, :-1]
        nbr[:-1, :-1] |= out[1:, 1:]
        nbr[1:, 1:] |= out[:-1, :-1]
        nbr[:-1, 1:] |= out[1:, :-1]
        nbr[1:, :-1] |= out[:-1, 1:]
        out = nbr
    return out


def _render_frame_xy(
    density_xy: np.ndarray,
    ghost_xy: Optional[np.ndarray],
    *,
    density_vmin: float,
    density_vmax: float,
    ghost_alpha: float,
    ghost_thr: Optional[float],
    ghost_global_max: Optional[float],
    ghost_gamma: float,
    ghost_boost: float,
    ghost_min_alpha: float,
    ghost_dilate: int,
    title: str,
    fig: plt.Figure,
    ax: plt.Axes,
    canvas: FigureCanvas,
    add_colorbar: bool,
    cbar_ax: Optional[plt.Axes],
):
    """Render a single XY density slice with optional ghost overlay.

    The frame is drawn to an Agg canvas and returned as uint8 RGB.
    """

    ax.clear()

    im = ax.imshow(
        density_xy,
        cmap="viridis",
        vmin=density_vmin,
        vmax=density_vmax,
        origin="lower",
        interpolation="nearest",
    )

    # Optional ghost overlay as magenta RGBA image.
    if ghost_xy is not None and ghost_alpha > 0.0:
        g = ghost_xy.astype(np.float32, copy=False)

        # Normalize against a *global* scale so appearance is consistent across z.
        # This avoids a common failure mode where each slice is normalized by its
        # own min/max and ghost becomes nearly invisible.
        if ghost_global_max is not None and ghost_global_max > 0:
            g01 = (g / float(ghost_global_max)).clip(0.0, 1.0)
        else:
            g01 = _normalize01(g)

        if ghost_thr is not None and ghost_global_max is not None:
            thr_val = float(ghost_thr) * float(ghost_global_max)
            g01 = np.where(g >= thr_val, g01, 0.0)

        # Optional dilation to expand ghost region (more vivid/visible).
        if ghost_dilate > 0:
            m = _dilate_bool(g01 > 0.0, int(ghost_dilate))
            g01 = np.where(m, g01, 0.0)

        if np.any(g01 > 0):
            # Boost + gamma to make small values visible.
            gg = (g01 * float(ghost_boost)).clip(0.0, 1.0)
            gg = np.power(gg, float(ghost_gamma))
            # Ensure a minimum alpha so thin ghosts still show up.
            gg = np.where(gg > 0, np.maximum(gg, float(ghost_min_alpha)), 0.0)

            rgba = np.zeros((g.shape[0], g.shape[1], 4), dtype=np.float32)
            rgba[..., 0] = 1.0  # R
            rgba[..., 2] = 1.0  # B
            rgba[..., 3] = (ghost_alpha * gg).clip(0.0, 1.0)  # A
            ax.imshow(rgba, origin="lower", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")

    if add_colorbar and cbar_ax is not None:
        cbar_ax.clear()
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label("Density")

    canvas.draw()

    # Robust pixel readback across matplotlib versions.
    try:
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        rgb = rgb.reshape((h, w, 3))
        return rgb
    except AttributeError:
        # Newer matplotlib prefers RGBA buffer APIs.
        buf = np.asarray(canvas.buffer_rgba())  # (H, W, 4), uint8
        return buf[:, :, :3].copy()


def _write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    """Write frames to MP4 using imageio if available; fallback to PNG sequence."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio  # type: ignore

        # Be explicit about the ffmpeg backend. Otherwise imageio may pick a TIFF
        # writer based on extension/installed plugins, which doesn't support fps.
        with imageio.get_writer(str(out_path), fps=fps, format="FFMPEG") as w:
            for fr in frames:
                w.append_data(fr)
        return
    except Exception as e:
        # Fallback: save PNG sequence so user still gets results.
        seq_dir = out_path.with_suffix("")
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            png_path = seq_dir / f"frame_{i:04d}.png"
            plt.imsave(png_path, fr)
        print(f"[warn] Failed to write video {out_path} ({e}). Saved PNG sequence to {seq_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save 2D slice videos (Z scan) from reconstruction.pt")
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to reconstruction.pt OR a directory containing reconstruction.pt",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing reconstruction.pt (alternative to positional input)",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video frames-per-second")
    parser.add_argument("--dpi", type=int, default=140, help="Render DPI")
    parser.add_argument(
        "--p99",
        type=float,
        default=99.0,
        help="Percentile for robust density vmax (consistent across videos)",
    )
    parser.add_argument(
        "--ghost-alpha",
        type=float,
        default=1.0,
        help="Max alpha for magenta ghost overlay",
    )
    parser.add_argument(
        "--ghost-dilate",
        type=int,
        default=2,
        help="Binary dilation radius (in pixels) to expand ghost regions for visibility",
    )
    parser.add_argument(
        "--ghost-gamma",
        type=float,
        default=0.5,
        help="Gamma applied to ghost intensity (values <1 make faint ghosts brighter)",
    )
    parser.add_argument(
        "--ghost-boost",
        type=float,
        default=6.0,
        help="Multiply ghost intensity before gamma (higher makes ghost more vivid)",
    )
    parser.add_argument(
        "--ghost-min-alpha",
        type=float,
        default=0.8,
        help="Minimum alpha applied wherever ghost is present (0 disables)",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        input_path = Path(args.input_dir)
    elif args.input:
        input_path = Path(args.input)
    else:
        recons = sorted(list(Path("result").glob("**/reconstruction.pt")))
        if not recons:
            print("No reconstruction.pt found in result/ and no input provided.")
            sys.exit(1)
        input_path = recons[-1]

    input_path = input_path.expanduser().resolve()
    pt_path = _resolve_reconstruction_path(input_path)
    run_dir = pt_path.parent

    vol = _load_reconstruction_volume(pt_path)
    if vol.ndim != 3:
        raise ValueError(f"Expected reconstruction volume to be 3D, got shape={vol.shape}")

    X, Y, Z = vol.shape
    print(f"Loaded reconstruction: shape={vol.shape}, dtype={vol.dtype}")

    density_vmin = 0.0
    density_vmax = _robust_vmax(vol, p=float(args.p99))

    ghost = _load_ghost_union(run_dir, (X, Y, Z))
    ghost_global_max = float(np.nanmax(ghost)) if ghost is not None and np.isfinite(ghost).any() else None
    if ghost is None:
        print("No ghost volume found (run_dir/ghost_volume/*.pt). Will generate density-only video.")

    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Mode -> ghost threshold fraction of global max.
    # Use None for "no ghost" and also for "full ghost" (no thresholding).
    modes = [
        ("no_ghost", None),
        ("ghost_full", None),
        ("ghost_thr025", 0.25),
        ("ghost_thr050", 0.50),
    ]

    # Create a single figure with a persistent cbar axis so colorbar scale is consistent.
    # We redraw the colorbar only on the first frame for each mode.
    fig = plt.Figure(figsize=(7.5, 6.5), dpi=int(args.dpi))
    canvas = FigureCanvas(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    for mode_name, thr in modes:
        if mode_name != "no_ghost" and ghost is None:
            # Skip overlay modes if ghost isn't available.
            continue

        frames: list[np.ndarray] = []
        for z in range(Z):
            density_xy = vol[:, :, z]

            # IMPORTANT: never overlay ghost in no_ghost mode.
            if mode_name == "no_ghost":
                ghost_xy = None
                ghost_thr = None
            else:
                ghost_xy = ghost[:, :, z] if ghost is not None else None
                ghost_thr = thr

            title = f"XY slice z={z}/{Z-1} | mode={mode_name}"
            frame = _render_frame_xy(
                density_xy,
                ghost_xy,
                density_vmin=density_vmin,
                density_vmax=density_vmax,
                ghost_alpha=float(args.ghost_alpha),
                ghost_thr=ghost_thr,
                ghost_global_max=ghost_global_max,
                ghost_gamma=float(args.ghost_gamma),
                ghost_boost=float(args.ghost_boost),
                ghost_min_alpha=float(args.ghost_min_alpha),
                ghost_dilate=int(args.ghost_dilate),
                title=title,
                fig=fig,
                ax=ax,
                canvas=canvas,
                add_colorbar=(z == 0),
                cbar_ax=cax,
            )
            frames.append(frame)

        out_path = viz_dir / f"density_xy_{mode_name}.mp4"
        _write_video(frames, out_path, fps=int(args.fps))
        print(f"Saved {out_path} ({len(frames)} frames)")

    plt.close(fig)


if __name__ == "__main__":
    main()
