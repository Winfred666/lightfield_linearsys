import numpy as np
import pyvista as pv
import torch
import argparse
from pathlib import Path
import sys
from typing import Optional


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
    return vol.float().numpy()


def _load_ghost_union(input_dir: Path, target_shape: tuple[int, int, int]) -> Optional[np.ndarray]:
    """Union ghost volumes under input_dir/ghost_volume/*.pt into a single float32 volume.

    Expected each *.pt to contain either a dict with key 'ghost'/'ghost_volume' or a raw tensor.
    Any values are accumulated (sum), yielding a density union.
    """
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
                # common case: driver_pair saves {'ghost': tensor, ...}; else first tensor-like.
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
        # Density union: sum (works as union-visibility; max would also be reasonable)
        union += arr

    return union

def _print_stats(name: str, a: np.ndarray) -> None:
    a = np.asarray(a)
    finite = np.isfinite(a)
    if not np.any(finite):
        print(f"{name}: no finite values (shape={a.shape}, dtype={a.dtype})")
        return
    af = a[finite]
    af64 = af.astype(np.float64, copy=False)
    print(
        f"{name} stats (finite only): "
        f"min={np.min(af64):.6g}, max={np.max(af64):.6g}, "
        f"mean={np.mean(af64):.6g}, median={np.median(af64):.6g} "
        f"(shape={a.shape}, dtype={a.dtype})"
    )

def _make_opacity_transfer(vmin: float, vmax: float, thr: float, ramp: float = 0.05):
    """Build a simple opacity transfer function."""
    x = np.linspace(vmin, vmax, 256, dtype=np.float32)
    width = float(ramp) * (vmax - vmin)
    width = max(width, 1e-12)
    op = np.zeros_like(x, dtype=np.float32)
    m = (x > thr) & (x < thr + width)
    op[m] = (x[m] - thr) / width
    op[x >= thr + width] = 1.0
    return op

def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--z_step", type=float, default=1.0, help="Z-step in um")
    parser.add_argument("--pixel_size", type=float, default=1.0, help="Pixel size in um")
    args = parser.parse_args()

    if args.input_dir is not None:
        input_path = Path(args.input_dir)
    elif args.input:
        input_path = Path(args.input)
    else:
        # Try to find the latest reconstruction
        recons = sorted(list(Path("result").glob("**/reconstruction.pt")))
        if not recons:
            print("No reconstruction.pt found in result/ and no input provided.")
            sys.exit(1)
        input_path = recons[-1]

    # Normalize to a reconstruction.pt file, but remember parent run dir.
    input_path = input_path.expanduser().resolve()
    pt_path = _resolve_reconstruction_path(input_path)
    run_dir = pt_path.parent
    
    print(f"Loading {pt_path}...")
    vol_np = _load_reconstruction_volume(pt_path)
    _print_stats("volume", vol_np)

    # PyVista grid
    grid = pv.ImageData()
    grid.dimensions = vol_np.shape # (X, Y, Z)
    grid.spacing = (args.pixel_size, args.pixel_size, args.z_step)
    
    nx, ny, nz = vol_np.shape
    grid.origin = (-(nx-1)/2.0 * args.pixel_size, -(ny-1)/2.0 * args.pixel_size, 0.0)
    grid.point_data["intensity"] = vol_np.ravel(order="F")

    # Stats for color limits and opacity
    finite_vals = vol_np[np.isfinite(vol_np)]
    if finite_vals.size == 0:
        print("No finite values in volume; cannot visualize.")
        sys.exit(1)

    # For light-field recon volumes, values are often near-0 with a heavy tail.
    # Using raw min/max makes the dynamic range too large and everything looks empty.
    # Use robust upper bound for both clim and opacity mapping.
    p99 = float(np.percentile(finite_vals, 99))
    vmax_vis = max(p99, 1e-12)

    # Clamp color range to [0, p99] by default so faint positive signal becomes visible.
    # Color range is user-controllable via sliders; start with [0, p99]
    clim_low0 = 0.0
    clim_high0 = vmax_vis

    # Opacity transfer will be built over [0, p99] as well.
    vmin, vmax = 0.0, vmax_vis

    # Default threshold: 0 means "show everything above 0".
    thr0 = 0.0

    # Optional ghost overlay (union of all ghost_volume/*.pt)
    ghost_np = _load_ghost_union(run_dir, tuple(vol_np.shape))
    if ghost_np is not None:
        _print_stats("ghost_union", ghost_np)
    ghost_finite = ghost_np[np.isfinite(ghost_np)] if ghost_np is not None else None
    ghost_p99 = (
        float(np.percentile(ghost_finite, 99))
        if ghost_np is not None and ghost_finite is not None and ghost_finite.size > 0
        else None
    )
    ghost_vmax = max(ghost_p99, 1e-12) if ghost_p99 is not None else None

    pl = pv.Plotter(window_size=(1200, 900))
    pl.set_background("white")

    # Keep a stable actor name so slider updates replace the existing volume.
    VOLUME_ACTOR_NAME = "vol"
    GHOST_ACTOR_NAME = "ghost"

    # Shared state for slider callbacks
    state = {
        "thr": float(thr0),
        "clim_low": float(clim_low0),
        "clim_high": float(clim_high0),
        "ghost_opacity": 0.35,
    }

    def _update_main_volume(*_args):
        thr = float(state["thr"])
        clim_low = float(state["clim_low"])
        clim_high = float(state["clim_high"])

        if clim_high <= clim_low:
            clim_high = clim_low + 1e-12
            state["clim_high"] = clim_high

        opacity_tf = _make_opacity_transfer(vmin, vmax, thr, ramp=0.02)
        # Remove old actor first (older PyVista versions don't always replace cleanly)
        try:
            pl.remove_actor(VOLUME_ACTOR_NAME)
        except Exception:
            pass
        pl.add_volume(
            grid,
            cmap="jet",
            opacity=opacity_tf,
            clim=(clim_low, clim_high),
            name=VOLUME_ACTOR_NAME,
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Intensity', 'vertical': True}
        )

    def _update_ghost_volume(*_args):
        if ghost_np is None or ghost_vmax is None:
            return
        op = float(state["ghost_opacity"])
        op = min(max(op, 0.0), 1.0)
        state["ghost_opacity"] = op

        # For the ghost overlay we keep a very simple ramp over [0, p99]
        opacity_tf = np.linspace(0.0, op, 256, dtype=np.float32)

        try:
            pl.remove_actor(GHOST_ACTOR_NAME)
        except Exception:
            pass

        pl.add_volume(
            ghost_grid,
            # PyVista expects a colormap name (string) or a list of string color names.
            # Using a single-color colormap isn't supported via raw RGB tuples here,
            # so we pick a built-in colormap with good contrast.
            cmap="magma",
            opacity=opacity_tf,
            clim=(0.0, ghost_vmax),
            name=GHOST_ACTOR_NAME,
            show_scalar_bar=True,
        )

    # Initial main volume
    _update_main_volume()

    # If present, create ghost grid and add it once
    if ghost_np is not None and ghost_vmax is not None:
        ghost_grid = pv.ImageData()
        ghost_grid.dimensions = ghost_np.shape
        ghost_grid.spacing = (args.pixel_size, args.pixel_size, args.z_step)
        ghost_grid.origin = grid.origin
        ghost_grid.point_data["intensity"] = ghost_np.ravel(order="F")
        _update_ghost_volume()

    # 1. Slider Widget for Transparency Threshold (main)
    pl.add_slider_widget(
        callback=lambda v: (state.__setitem__("thr", float(v)), _update_main_volume()),
        rng=(vmin, vmax),
        value=thr0,
        title="Transparency Threshold",
        pointa=(0.05, 0.1),
        pointb=(0.35, 0.1),
        style='modern',
        color='black'
    )

    # 1b. Two sliders for clim low/high (main)
    pl.add_slider_widget(
        callback=lambda v: (state.__setitem__("clim_low", float(v)), _update_main_volume()),
        rng=(vmin, vmax),
        value=clim_low0,
        title="CLim Low",
        pointa=(0.40, 0.1),
        pointb=(0.70, 0.1),
        style='modern',
        color='black'
    )
    pl.add_slider_widget(
        callback=lambda v: (state.__setitem__("clim_high", float(v)), _update_main_volume()),
        rng=(vmin, vmax),
        value=clim_high0,
        title="CLim High",
        pointa=(0.75, 0.1),
        pointb=(0.98, 0.1),
        style='modern',
        color='black'
    )

    # 1c. Ghost overlay opacity slider (if available)
    if ghost_np is not None and ghost_vmax is not None:
        pl.add_slider_widget(
            callback=lambda v: (state.__setitem__("ghost_opacity", float(v)), _update_ghost_volume()),
            rng=(0.0, 1.0),
            value=state["ghost_opacity"],
            title="Ghost Opacity",
            pointa=(0.05, 0.03),
            pointb=(0.35, 0.03),
            style='modern',
            color='black'
        )

    # 2. Clip Plane Widget
    # This is a very visible UI widget that allows interactive slicing
    # Clip plane widget for interactive slicing.
    # NOTE: API varies across PyVista versions; keep kwargs minimal to avoid forwarding
    # unexpected kwargs into add_volume (which can raise TypeError).
    pl.add_volume_clip_plane(
        grid,
        normal='-x',
        assign_to_axis='x',
        origin=grid.center,
    )

    # 3. Add Outline and Axes
    pl.add_mesh(grid.outline(), color="black")
    pl.add_axes()

    # 4. Bounds with labels
    pl.show_bounds(
        grid=True,
        location='outer',
        ticks='both',
        n_xlabels=3,
        n_ylabels=3,
        n_zlabels=3,
        xtitle='X (um)',
        ytitle='Y (um)',
        ztitle='Z (um)',
        color='black'
    )

    print("Opening PyVista window... (Interactive widgets enabled)")
    pl.show()

if __name__ == "__main__":
    main()
