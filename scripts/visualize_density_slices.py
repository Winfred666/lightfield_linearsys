#!/usr/bin/env python3

"""
Visualize reconstruction results and reprojection error.

This script:
1. Loads a reconstruction volume (reconstruction.pt).
2. Loads projection pairs (pair_*.h5) from a data directory.
3. Reprojects the reconstruction using the lightfield 'A' from the pairs.
4. Compares the reprojected image with the target image 'b'.
5. Generates comparison plots (MSE/PSNR) in viz/reprojection/.
6. Generates a grid of Z-slices of the reconstruction volume in viz/.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import logging
import re

from LF_linearsys.io.data_postclean import compute_valid_z_indices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _resolve_reconstruction_path(p: Path) -> Path:
    """Accept either a reconstruction.pt file or a directory containing one."""
    if p.is_dir():
        cand = p / "reconstruction.pt"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"No reconstruction.pt found in directory: {p}")
    return p

def _load_reconstruction_volume(pt_path: Path) -> torch.Tensor:
    logger.info(f"Loading reconstruction from {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, dict) and "reconstruction" in data:
        vol = data["reconstruction"]
    else:
        vol = data
    return vol.detach().cpu().float()

def _load_pair(h5_path: Path):
    with h5py.File(h5_path, 'r') as f:
        # A is (X, Y, Z), b is (Y, X)
        A = torch.from_numpy(f['A'][:]).float()
        b = torch.from_numpy(f['b'][:]).float()
    return A, b

def project_and_compare(vol: torch.Tensor, A: torch.Tensor, b: torch.Tensor):
    # vol: (X, Y, Z)
    # A: (X, Y, Z)
    # b: (Y, X)
    
    # b_pred(y, x) = sum_z ( A(x, y, z) * vol(x, y, z) )
    # Element-wise mult and sum over Z -> (X, Y)
    
    # Ensure shapes match (handle potential mismatches gracefully)
    if A.shape != vol.shape:
        logger.warning(f"Shape mismatch: A {A.shape} vs vol {vol.shape}. Truncating to common size.")
        sx = min(A.shape[0], vol.shape[0])
        sy = min(A.shape[1], vol.shape[1])
        sz = min(A.shape[2], vol.shape[2])
        A = A[:sx, :sy, :sz]
        vol_use = vol[:sx, :sy, :sz]
    else:
        vol_use = vol

    b_pred_xy = torch.sum(A * vol_use, dim=2) # (X, Y)
    b_pred = b_pred_xy.T # (Y, X)
    
    # Metrics
    diff = b - b_pred
    mse = torch.mean(diff**2).item()
    
    # Robust dynamic range for PSNR
    data_max = b.max().item()
    if data_max <= 0: 
        data_max = 1.0 # Avoid div by zero or log of zero if image is all zeros
    
    if mse <= 1e-12:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(data_max / np.sqrt(mse))
        
    return b_pred, mse, psnr

def visualize_reprojection(b, b_pred, mse, psnr, out_path, pair_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Shared color scale for fair comparison
    # We use robust min/max to avoid outliers skewing the plot
    b_np = b.numpy()
    pred_np = b_pred.numpy()
    
    vmin = min(np.percentile(b_np, 1), np.percentile(pred_np, 1))
    vmax = max(np.percentile(b_np, 99), np.percentile(pred_np, 99))
    
    # Ensure sane range if image is empty
    if vmax <= vmin:
        vmin, vmax = 0, 1

    im0 = axes[0].imshow(b_np, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"GT {pair_name} (b)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reprojection (A@x)\nMSE={mse:.2e}, PSNR={psnr:.2f} dB")
    plt.colorbar(im1, ax=axes[1])
    
    plt.suptitle(f"Reprojection Comparison: {pair_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def visualize_slices(vol, out_path):
    nz = vol.shape[2]
    num_slices = 25
    indices = np.linspace(0, nz - 1, num_slices, dtype=int)

    ncols = int(np.ceil(np.sqrt(num_slices)))
    nrows = int(np.ceil(num_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()
    
    # Robust range for slices
    vol_flat = vol.numpy().ravel()
    if vol_flat.size > 0:
        p99 = np.percentile(vol_flat, 99)
        vmax = max(p99, 1e-6)
    else:
        vmax = 1.0
        
    im = None
    for i, idx in enumerate(indices):
        # Slice: vol[:, :, idx] -> (X, Y)
        # Transpose to (Y, X) for imshow standard orientation
        slice_data = vol[:, :, idx].numpy().T
        
        im = axes[i].imshow(slice_data, cmap='viridis', vmin=0, vmax=vmax)
        axes[i].set_title(f"Z={idx}")
        axes[i].axis('off')

    # Hide unused axes
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')
        
    if im is not None:
        # Add shared colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Density')
        
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def visualize_volume_rendering(vol: torch.Tensor, out_path: Path):
    """
    Simulates a volume rendering from the 'side' (along X axis) 
    by projecting density.
    Result: Left->Right is Z-axis, Bottom->Top is Y-axis.
    """
    # vol is (X, Y, Z).
    # Project along X (dim 0) -> Result shape (Y, Z).
    # Use SUM projection (same spirit as project_and_compare) rather than MIP.
    # In imshow, dim 0 is vertical (Y), dim 1 is horizontal (Z).
    
    vol_np = vol.numpy()
    
    # Project along X (dim 0) -> (Y, Z)
    proj = np.sum(vol_np, axis=0)

    # Keep physical aspect ratio (no axis distortion).
    # Data is (Y, Z) where Z is width and Y is height.
    # Example you gave: Z~400, Y~1024 -> tall image. We size accordingly.
    h, w = proj.shape  # (Y, Z)
    w_in = 6.0
    h_in = max(3.0, w_in * (h / max(w, 1)))
    fig, ax = plt.subplots(figsize=(w_in, h_in))

    # Do NOT normalize to 0-1. Show density values.
    # Use robust range for readability but keep the units in the colorbar.
    if proj.size > 0:
        vmin = float(np.percentile(proj, 1))
        vmax = float(np.percentile(proj, 99))
        if vmax <= vmin:
            vmin, vmax = float(np.min(proj)), float(np.max(proj))
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(
        proj,
        cmap='viridis',
        aspect='equal',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title("Side View (Sum projection along X)\nLeft->Right is Z-axis")
    ax.set_xlabel("Z (Depth)")
    ax.set_ylabel("Y (Vertical)")
    fig.colorbar(im, ax=ax, label='Projected density (sum over X)')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def _robust_vmax(vol: np.ndarray, p: float = 99.0) -> float:
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return 1e-12
    v = float(np.percentile(finite, p))
    return max(v, 1e-12)

def _render_frame_xy(
    density_xy: np.ndarray,
    *,
    density_vmin: float,
    density_vmax: float,
    title: str,
    fig: plt.Figure,
    ax: plt.Axes,
    canvas: FigureCanvas,
    add_colorbar: bool,
    cbar_ax: plt.Axes,
):
    """Render a single XY density slice."""
    ax.clear()

    im = ax.imshow(
        density_xy,
        cmap="viridis",
        vmin=density_vmin,
        vmax=density_vmax,
        origin="lower",
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")

    if add_colorbar and cbar_ax is not None:
        cbar_ax.clear()
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label("Density")

    canvas.draw()

    # Robust pixel readback
    try:
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        rgb = rgb.reshape((h, w, 3))
        return rgb
    except AttributeError:
        buf = np.asarray(canvas.buffer_rgba())
        return buf[:, :, :3].copy()

def _write_video(frames: list[np.ndarray], out_path: Path, fps: int = 20) -> None:
    """Write frames to MP4 using imageio if available; fallback to PNG sequence."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(out_path), fps=fps, format="FFMPEG") as w:
            for fr in frames:
                w.append_data(fr)
        return
    except Exception as e:
        seq_dir = out_path.with_suffix("")
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            png_path = seq_dir / f"frame_{i:04d}.png"
            plt.imsave(png_path, fr)
        logger.warning(f"Failed to write video {out_path} ({e}). Saved PNG sequence to {seq_dir}.")

def visualize_z_scan_video(vol: torch.Tensor, out_path: Path):
    """Generates a Z-scan video of the reconstruction volume."""
    vol_np = vol.numpy()
    X, Y, Z = vol_np.shape
    
    density_vmin = 0.0
    density_vmax = _robust_vmax(vol_np, p=99.0)

    fig = plt.Figure(figsize=(7.5, 6.5), dpi=140)
    canvas = FigureCanvas(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    frames = []
    logger.info(f"Rendering {Z} frames for Z-scan video...")
    for z in range(Z):
        density_xy = vol_np[:, :, z]
        title = f"XY slice z={z}/{Z-1}"
        
        frame = _render_frame_xy(
            density_xy,
            density_vmin=density_vmin,
            density_vmax=density_vmax,
            title=title,
            fig=fig,
            ax=ax,
            canvas=canvas,
            add_colorbar=(z == 0),
            cbar_ax=cax,
        )
        frames.append(frame)

    _write_video(frames, out_path)
    plt.close(fig)
    logger.info(f"Saved Z-scan video to {out_path}")

def numerical_sort_key(p: Path):
    # Extract number from filename like pair_12.h5
    numbers = re.findall(r'\d+', p.name)
    if numbers:
        return int(numbers[-1])
    return p.name

def main():
    parser = argparse.ArgumentParser(description="Visualize density slices and reprojection error.")
    parser.add_argument("input", nargs="?", help="Path to reconstruction.pt or directory")
    parser.add_argument("--data-dir", default=None, help="Directory containing pair_*.h5 files (Processed mode)")
    
    # Raw Data Mode
    parser.add_argument("--raw-A-dir", default=None, help="Raw volume directory (e.g. data/raw/lightsheet_vol_6.9)")
    parser.add_argument("--raw-b-dir", default=None, help="Raw image directory (e.g. data/raw/20um_imgs)")
    parser.add_argument("--downsampling-rate", type=float, default=0.125, help="Downsampling rate for raw mode")
    parser.add_argument("--scale-factor", type=float, default=8.0, help="Scale factor for raw mode")
    
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <run_dir>/viz)")
    parser.add_argument("--stride-pairs", type=int, default=1, help="Stride for processing pair files (default: 1)")
    
    args = parser.parse_args()
    
    if args.input:
        input_path = Path(args.input)
    else:
        # Auto-detect latest reconstruction
        recons = sorted(list(Path("result").glob("**/reconstruction.pt")))
        if not recons:
            print("No reconstruction.pt found in result/ and no input provided.")
            sys.exit(1)
        input_path = recons[-1]
        
    pt_path = _resolve_reconstruction_path(input_path.expanduser().resolve())
    run_dir = pt_path.parent
    
    if args.output_dir:
        viz_dir = Path(args.output_dir)
    else:
        viz_dir = run_dir / "viz"
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    reproj_dir = viz_dir / "reprojection"
    reproj_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reconstruction
    vol = _load_reconstruction_volume(pt_path)
    logger.info(f"Reconstruction loaded. Shape: {vol.shape}, Max: {vol.max():.4f}")
    
    # 1. Reprojection Analysis
    
    # Check for Raw Mode
    raw_mode = (args.raw_A_dir is not None and args.raw_b_dir is not None)
    
    if raw_mode:
        from LF_linearsys.io.preprocess_pair import preprocess_one_pair
        from LF_linearsys.io.raw_pairs import find_raw_pairs
        
        logger.info("Using Raw Data Mode for reprojection.")
        pairs = find_raw_pairs(args.raw_A_dir, args.raw_b_dir)
        
        if not pairs:
            logger.warning(f"No matching raw pairs found in {args.raw_A_dir}")
        else:
            pairs = pairs[::args.stride_pairs]
            logger.info(f"Processing {len(pairs)} raw pairs (stride={args.stride_pairs})...")
            
            for p in pairs:
                try:
                    logger.info(f"Processing Raw Pair Index {p.idx}...")
                    
                    # On-the-fly preprocess
                    A, b = preprocess_one_pair(
                        vol_path=p.vol_path,
                        img_path=p.img_path,
                        downsampling_rate=args.downsampling_rate,
                        scale_factor=args.scale_factor,
                        device=torch.device("cpu") # Keep on CPU for visualization
                    )

                    # Annotate which Z slices are actually active in A.
                    # This helps interpret reprojection quality when the system has a limited Z support.
                    valid_z = compute_valid_z_indices(A, threshold_A=0.01)
                    if valid_z.numel() > 0:
                        z_min = int(valid_z.min().item())
                        z_max = int(valid_z.max().item())
                        pair_label = f"Raw_Pair_{p.idx} [{z_min},{z_max}]"
                    else:
                        pair_label = f"Raw_Pair_{p.idx} [empty]"
                    
                    b_pred, mse, psnr = project_and_compare(vol, A, b)
                    out_name = f"target_image_compare_{p.idx}.png"
                    visualize_reprojection(b, b_pred, mse, psnr, reproj_dir / out_name, pair_label)
                    
                except Exception as e:
                    logger.error(f"Failed to process pair {p.idx}: {e}")
                    
    elif args.data_dir:
        # Processed HDF5 Mode
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            sys.exit(1)

        pair_files = sorted(list(data_dir.glob("pair_*.h5")), key=numerical_sort_key)
        
        if not pair_files:
            logger.warning(f"No pair_*.h5 files found in {data_dir}")
        else:
            pair_files = pair_files[::args.stride_pairs]
            logger.info(f"Processing {len(pair_files)} pair files (stride={args.stride_pairs})...")
            
            # Process all pairs
            for p_file in pair_files:
                try:
                    # Extract simple name/number
                    stem = p_file.stem # pair_123
                    parts = stem.split('_')
                    num = parts[-1] if len(parts) > 1 else "0"
                    
                    logger.info(f"Processing {stem}...")
                    A, b = _load_pair(p_file)

                    valid_z = compute_valid_z_indices(A, threshold_A=0.01)
                    if valid_z.numel() > 0:
                        z_min = int(valid_z.min().item())
                        z_max = int(valid_z.max().item())
                        pair_label = f"{stem} [{z_min},{z_max}]"
                    else:
                        pair_label = f"{stem} [empty]"

                    b_pred, mse, psnr = project_and_compare(vol, A, b)
                    
                    out_name = f"target_image_compare_{num}.png"
                    visualize_reprojection(b, b_pred, mse, psnr, reproj_dir / out_name, pair_label)
                    
                except Exception as e:
                    logger.error(f"Failed to process {p_file}: {e}")
    else:
        logger.error("No data source provided. Use --data-dir (H5) or --raw-A-dir/--raw-b-dir (Raw).")
        sys.exit(1)

    # 2. Volume Slices
    logger.info("Generating volume slices...")
    visualize_slices(vol, viz_dir / "reconstruction_slices.png")
    
    # 3. Side View Volume Render
    logger.info("Generating side view volume render...")
    visualize_volume_rendering(vol, viz_dir / "volume_render_side.png")

    # 4. Z-Scan Video
    logger.info("Generating Z-scan video...")
    visualize_z_scan_video(vol, viz_dir / "reconstruction_z_scan.mp4")
    
    logger.info(f"Visualization complete. Results in {viz_dir}")

if __name__ == "__main__":
    main()
