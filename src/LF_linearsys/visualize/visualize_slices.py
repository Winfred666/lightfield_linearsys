import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys
import logging
import yaml


def setup_logging(output_dir):
    log_path = output_dir / "visualization.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def normalize_img(img):
    if isinstance(img, torch.Tensor):
        img = img.float().cpu().numpy()
    img_min, img_max = img.min(), img.max()
    if img_max == img_min:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

def normalize_img_percentile(img, p_low=1.0, p_high=99.0):
    """Robust normalization using percentiles (good for sparse / heavy-tailed data)."""
    if isinstance(img, torch.Tensor):
        img = img.float().cpu().numpy()
    img = np.asarray(img)
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        return np.zeros_like(img)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def overlay_nonzero_red(gray01, mask, alpha=0.9):
    """Overlay a boolean mask as red on top of a grayscale image (HxW -> HxWx3)."""
    gray01 = np.asarray(gray01)
    mask = np.asarray(mask).astype(bool)
    rgb = np.stack([gray01, gray01, gray01], axis=-1)
    if mask.any():
        rgb[mask, 0] = 1.0
        rgb[mask, 1] = rgb[mask, 1] * (1.0 - alpha)
        rgb[mask, 2] = rgb[mask, 2] * (1.0 - alpha)
    return rgb


def _downsample_2d(arr2d, factor: int):
    """Fast spatial downsample by striding (no interpolation)."""
    if factor is None or factor <= 1:
        return arr2d
    return arr2d[::factor, ::factor]


def visualize_reconstruction(
    result_path,
    output_dir,
    highlight_nonzero: bool = True,
    nonzero_eps: float = 0.0,
    norm_mode: str = "percentile",
    video_stride: int = 1,
    video_downsample: int = 2,
    video_max_frames: int | None = 200,
    video_fps: int = 10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info(f"Loading result from {result_path}")
    data = torch.load(result_path, map_location='cpu')
    
    if isinstance(data, dict):
        vol = data['reconstruction']
        history = data.get('history', {})
        valid_Ax = data.get('valid_Ax')
        valid_b = data.get('valid_b')
    else:
        # Legacy format (just tensor)
        vol = data
        history = {}
        valid_Ax = None
        valid_b = None
        
    logger.info(f"Volume Shape: {vol.shape}")

    # 1. Visualize Convergence
    if history:
        logger.info("Plotting convergence...")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(history['loss'], color=color, label='Total Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Residual Norm ||Ax-b||', color=color)  
        ax2.plot(history['residual_norm'], color=color, linestyle='--', label='Residual')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Solver Convergence")
        fig.tight_layout()
        plt.savefig(output_dir / "convergence.png")
        plt.close()

    # 2. Visualize Ax vs b Scatter
    if valid_Ax is not None and valid_b is not None:
        logger.info("Plotting Ax vs b scatter (subsampled)...")
        # Subsample if too large
        n_points = valid_Ax.numel()
        if n_points > 10000:
            indices = torch.randperm(n_points)[:10000]
            va = valid_Ax[indices].float().numpy()
            vb = valid_b[indices].float().numpy()
        else:
            va = valid_Ax.float().numpy()
            vb = valid_b.float().numpy()
            
        plt.figure(figsize=(8, 8))
        plt.scatter(vb, va, alpha=0.1, s=1)
        
        # Ideal line
        min_val = min(vb.min(), va.min())
        max_val = max(vb.max(), va.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal x=y')
        
        plt.xlabel("Measured (b)")
        plt.ylabel("Projected (Ax)")
        plt.title("Fidelity Check: Ax vs b")
        plt.legend()
        plt.savefig(output_dir / "fidelity_scatter.png")
        plt.close()

    # 3. Volume Slices (Grid)
    logger.info("Generating Volume Slices...")
    nz = vol.shape[2]
    num_slices = 9
    indices = np.linspace(0, nz-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        slice_data = vol[:, :, idx].float().numpy().T 
        if norm_mode == "minmax":
            im_norm = normalize_img(slice_data)
        else:
            im_norm = normalize_img_percentile(slice_data, p_low=1.0, p_high=99.0)

        if highlight_nonzero:
            mask = slice_data > nonzero_eps
            rgb = overlay_nonzero_red(im_norm, mask)
            axes[i].imshow(rgb)
            axes[i].set_title(f"Z-Slice {idx} (nonzero>{nonzero_eps:g})")
        else:
            axes[i].imshow(im_norm, cmap='inferno')
            axes[i].set_title(f"Z-Slice {idx}")

        # Keep axes visible for alignment debugging (do not hide ticks/axis).
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_slices.png")
    plt.close()

    # 4. Video (Z-Scan)
    logger.info("Generating Volume Video...")
    fig_vid, ax_vid = plt.subplots(figsize=(8, 8))

    stride = max(1, int(video_stride))
    frames = list(range(0, nz, stride))
    if video_max_frames is not None and len(frames) > int(video_max_frames):
        frames = frames[: int(video_max_frames)]

    if not frames:
        logger.warning("No frames selected for video; skipping video export.")
        plt.close(fig_vid)
        logger.info("Done.")
        return

    first_slice = vol[:, :, frames[0]].float().numpy().T
    first_slice = _downsample_2d(first_slice, video_downsample)

    if norm_mode == "minmax":
        first_norm = normalize_img(first_slice)
    else:
        first_norm = normalize_img_percentile(first_slice, p_low=1.0, p_high=99.0)

    if highlight_nonzero:
        first_mask = first_slice > nonzero_eps
        first_rgb = overlay_nonzero_red(first_norm, first_mask)
        im_display = ax_vid.imshow(first_rgb, animated=True)
    else:
        im_display = ax_vid.imshow(first_norm, cmap='inferno', animated=True)
    ax_vid.set_title(f"Z-Slice Video")
    # Keep axis visible (helps debugging orientation/alignment)
    ax_vid.set_xlabel("X")
    ax_vid.set_ylabel("Y")

    def update(frame):
        slice_data = vol[:, :, frame].float().numpy().T
        slice_data = _downsample_2d(slice_data, video_downsample)

        if norm_mode == "minmax":
            im_norm = normalize_img(slice_data)
        else:
            im_norm = normalize_img_percentile(slice_data, p_low=1.0, p_high=99.0)

        if highlight_nonzero:
            mask = slice_data > nonzero_eps
            rgb = overlay_nonzero_red(im_norm, mask)
            im_display.set_data(rgb)
        else:
            im_display.set_data(im_norm)

        ax_vid.set_title(f"Z-Slice {frame}")
        return [im_display]

    logger.info(
        f"Video frames: {len(frames)} (nz={nz}, stride={stride}, max_frames={video_max_frames}, downsample={video_downsample}x)"
    )
    ani = FuncAnimation(fig_vid, update, frames=frames, blit=True)
    try:
        ani.save(output_dir / "reconstruction_scan.mp4", writer='ffmpeg', fps=video_fps)
    except:
        logger.warning("ffmpeg not found, saving as gif")
        ani.save(output_dir / "reconstruction_scan.gif", writer='pillow', fps=video_fps)
        
    plt.close(fig_vid)
    logger.info("Done.")