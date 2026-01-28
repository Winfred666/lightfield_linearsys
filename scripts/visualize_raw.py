import torch
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys
import logging
from datetime import datetime
import argparse

# thresholds = [1.0, 0.9, 0.7, 0.6, 0.4, 0.2, 0.1, 1e-2]
thresholds = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.9, 0.8, 0.7]


def setup_logging(output_dir):
    log_path = output_dir / "analysis.log"
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
    """Normalize image to 0-1 range for visualization."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img_min, img_max = img.min(), img.max()
    if img_max == img_min:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

def normalize_img_percentile(img, p_low=1.0, p_high=99.0):
    """Robust normalization using percentiles (helps when most values are near 0)."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = np.asarray(img).copy()
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        return np.zeros_like(img)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

def overlay_nonzero_red(gray01, mask, alpha=0.9):
    """Overlay a boolean mask as red on top of a grayscale image.

    Args:
        gray01: 2D array normalized to [0, 1].
        mask: 2D boolean array, same shape.
        alpha: overlay strength for masked pixels.
    Returns:
        HxWx3 RGB image in [0, 1].
    """
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


def log_hist_edges_from_data(
    x: np.ndarray,
    decades: int = 6,
    include_zero: bool = True,
) -> np.ndarray:
    """
    Build log10-spaced histogram edges based on data range.
    - Includes explicit zero bin edge
    - Uses smallest positive value as lower bound
    - Uses max(|x|) as upper bound
    - Guarantees strictly increasing edges
    """
    x = np.asarray(x).ravel()
    if x.size == 0:
        return np.array([0.0, 1.0])

    x_abs = np.abs(x)
    x_max = float(np.nanmax(x_abs))
    if not np.isfinite(x_max) or x_max <= 0.0:
        return np.array([0.0, 1.0])

    pos = x_abs[x_abs > 0]
    if pos.size == 0:
        return np.array([0.0, x_max])

    x_min = float(np.nanmin(pos))
    if x_min <= 0.0:
        x_min = x_max

    # Compute decade bounds
    lo = np.floor(np.log10(x_min))
    hi = np.ceil(np.log10(x_max))

    bins_per_decade = max(1, int(decades))
    n_bins = max(2, int((hi - lo) * bins_per_decade))

    edges = np.logspace(lo, hi, num=n_bins, base=10)

    # Ensure numerical strict monotonicity
    edges = np.unique(edges)

    if include_zero and edges[0] > 0:
        edges = np.concatenate(([0.0], edges))

    return edges



def abs_values(arr: np.ndarray) -> np.ndarray:
    """Return absolute values (used for magnitude histograms)."""
    return np.abs(np.asarray(arr))


def robust_range_from_data(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> tuple[float, float]:
    """Return a robust (vmin, vmax) based on percentiles.

    Falls back to min/max if percentiles collapse.
    """
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
    return lo, hi


def annotate_histogram_counts(
    ax: plt.Axes,
    *,
    min_count: int = 1,
    max_labels: int = 30,
    min_x_gap: float = 0.03,
) -> None:
    """
    Write integer counts above histogram bars in a sparse, readable way.

    Strategy:
      - Prefer tallest bars
      - Enforce minimum x-distance between labels
      - Stagger vertical offsets
    """
    patches = list(ax.patches)
    if not patches:
        return

    # Extract bar info
    bars = []
    for p in patches:
        h = float(p.get_height())
        if h >= min_count:
            x = float(p.get_x() + p.get_width() / 2.0)
            bars.append((x, h, p))

    if not bars:
        return

    # Sort by descending height (importance)
    bars.sort(key=lambda t: t[1], reverse=True)

    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    min_dx = min_x_gap * x_range

    placed_x = []
    label_count = 0

    for i, (x, h, p) in enumerate(bars):
        if label_count >= max_labels:
            break

        # Enforce horizontal spacing
        if any(abs(x - px) < min_dx for px in placed_x):
            continue

        # Vertical staggering
        y_offset = 2 + (label_count % 3) * 6

        ax.annotate(
            f"{int(round(h))}",
            (x, h),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            xytext=(0, y_offset),
            textcoords="offset points",
            clip_on=True,
        )

        placed_x.append(x)
        label_count += 1

def visualize(
    file_path,
    args=None, # Pass full args object for flexibility
    norm_mode="percentile",
    video_downsample: int = 1,
    video_fps: int = 10,
    hist_log_bins: bool = True,
    hist_decades: int = 6,
):
    global thresholds
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"result/visualize_test/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Starting visualization for {file_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load Data
    try:
        logger.info("Loading data... (this might take a while for large files)")
        
        # Check if raw mode arguments are provided
        raw_mode = False
        if hasattr(args, 'input_dir') and args.input_dir and hasattr(args, 'img_dir') and args.img_dir:
            raw_mode = True
            
        if raw_mode:
             from LF_linearsys.io.preprocess_pair import preprocess_one_pair
             from LF_linearsys.io.raw_pairs import find_raw_pairs
             
             pairs = find_raw_pairs(args.input_dir, args.img_dir)
             target_pair = None
             
             # Try to match the specific file requested, or default to the first one
             target_idx = 1
             if args.file_path and "pair_" in args.file_path:
                 # Try to extract index from filename like pair_2.h5 or similar
                 import re
                 match = re.search(r'(\d+)', Path(args.file_path).name)
                 if match:
                     target_idx = int(match.group(1))
            
             # Find the matching pair
             for p in pairs:
                 if p.idx == target_idx:
                     target_pair = p
                     break
             
             if not target_pair and pairs:
                 logger.warning(f"Pair index {target_idx} not found, defaulting to first available pair {pairs[0].idx}")
                 target_pair = pairs[0]
                 
             if not target_pair:
                 logger.error("No valid raw pairs found.")
                 return

             logger.info(f"Processing Raw Pair Index {target_pair.idx} from {target_pair.vol_path} and {target_pair.img_path}")
             
             vol, img = preprocess_one_pair(
                 vol_path=target_pair.vol_path,
                 img_path=target_pair.img_path,
                 downsampling_rate=args.downsampling_rate,
                 scale_factor=args.scale_factor,
                 device=torch.device("cpu") # Visualize on CPU
             )
        else:
            # Legacy/Direct file mode
            file_path_obj = Path(file_path)
            if file_path_obj.suffix == '.h5':
                with h5py.File(file_path_obj, 'r') as f:
                    logger.info("Reading HDF5 datasets 'A' and 'b'...")
                    # Read into memory
                    vol = torch.from_numpy(f['A'][:])
                    img = torch.from_numpy(f['b'][:])
            else:
                data = torch.load(file_path, map_location='cpu')
                
                if isinstance(data, dict):
                    vol = data.get('A')
                    img = data.get('b')
                else:
                    # Fallback if structure is different
                    vol = data[0]
                    img = data[1]
            
        logger.info(f"Volume Shape: {vol.shape}, Type: {vol.dtype}")
        logger.info(f"Image Shape: {img.shape}, Type: {img.dtype}")
        
    except Exception as e:
        logger.error(f"Failed to load file or process raw data: {e}", exc_info=True)
        return

    # Convert to numpy for plotting (subsample volume for histograms to save memory/time)
    logger.info("Converting to numpy...")
    
    # Optional: Clip image values to [0, 1] as requested
    # img = torch.clamp(img, 0.0, 1.0)
    
    img_np = img.numpy()

    # Log basic stats for volume before any visualization normalization.
    # Use a subsample for speed/memory (volume can be huge).
    vol_np = vol.numpy()
    vol_np_sample = vol_np[::4, ::4, ::4]
    vol_flat = np.asarray(vol_np_sample).ravel()
    if vol_flat.size == 0:
        logger.warning("Volume is empty; cannot compute stats.")
    else:
        vol_min = float(np.min(vol_flat))
        vol_max = float(np.max(vol_flat))
        vol_mean = float(np.mean(vol_flat))
        vol_median = float(np.median(vol_flat))
        logger.info(
            "Volume stats (A) [subsample x4]: min=%.8g max=%.8g median=%.8g mean=%.8g",
            vol_min,
            vol_max,
            vol_median,
            vol_mean,
        )

    # Log basic stats for target image before normalization
    img_np_flat = np.asarray(img_np).ravel()
    if img_np_flat.size == 0:
        logger.warning("Target image is empty; cannot compute stats.")
        img_min, img_max, img_median, img_mean = 0, 0, 0, 0
    else:
        img_min = float(np.min(img_np_flat))
        img_max = float(np.max(img_np_flat))
        img_mean = float(np.mean(img_np_flat))
        img_median = float(np.median(img_np_flat))  # median of values
        logger.info(
            "Target image stats (pre-norm): min=%.8g max=%.8g median=%.8g mean=%.8g",
            img_min, img_max, img_median, img_mean
        )
    
    # --- 1. Histograms ---
    logger.info("Generating Histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if hist_log_bins:
        img_flat = np.asarray(img_np).ravel()
        edges = log_hist_edges_from_data(img_flat, decades=hist_decades, include_zero=True)
        img_hist = abs_values(img_flat)
        axes[0].hist(img_hist, bins=edges, color='blue', alpha=0.7)
        axes[0].set_xscale('log')
        axes[0].set_title(f"Target Image (b) Histogram (log bins, |x|)\nmin={img_min:.2g}, max={img_max:.2g}, mean={img_mean:.2g}, median={img_median:.2g}")
        axes[0].set_xlabel("|Intensity| (raw magnitude)")
        axes[0].set_ylabel("Count")
        # start at the first positive edge; zeros are still counted in the [0, edge1] bin
        if edges.size >= 3:
            axes[0].set_xlim(edges[1], edges[-1])
        annotate_histogram_counts(axes[0], min_count=1)
    else:
        # Linear histogram
        axes[0].hist(img_np.ravel(), bins=100, color='blue', alpha=0.7)
        axes[0].set_title(f"Target Image (b) Histogram\nmin={img_min:.2g}, max={img_max:.2g}, mean={img_mean:.2g}, median={img_median:.2g}")
        axes[0].set_xlabel("Intensity")
        axes[0].set_ylabel("Count")
    annotate_histogram_counts(axes[0], min_count=1)
    
    # Volume Histogram (Subsampled)
    vol_sample_np = vol_np_sample  # Stride 4 in each dim -> 1/64th size
    
    if hist_log_bins:
        # Use true data range bins so magnitudes are meaningful.
        edges = log_hist_edges_from_data(vol_sample_np.ravel(), decades=hist_decades, include_zero=True)
        vol_hist = abs_values(vol_sample_np.ravel())
        axes[1].hist(vol_hist, bins=edges, color='green', alpha=0.7)
        axes[1].set_xscale('log')
        axes[1].set_title(f"Volume (A) Histogram (Subsampled, log bins from data, |x|)")
        axes[1].set_xlabel("|Intensity| (raw magnitude)")
        axes[1].set_ylabel("Count")
        if edges.size >= 3:
            axes[1].set_xlim(edges[1], edges[-1])
        annotate_histogram_counts(axes[1], min_count=1)
    else:
        axes[1].hist(vol_sample_np.ravel(), bins=100, color='green', alpha=0.7)
        axes[1].set_title("Volume (A) Histogram (Subsampled)")
        axes[1].set_xlabel("Intensity")
        axes[1].set_ylabel("Count")
        annotate_histogram_counts(axes[1], min_count=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "histograms.png")
    plt.close()
    logger.info("Saved histograms.png")

    # --- 2. Threshold previews (absolute thresholds; hard-coded list) ---
    logger.info("Generating threshold previews (absolute thresholds; hard-coded list)...")
    
    threshold_output_dir = output_dir / "target_image_threshold_previews"
    threshold_output_dir.mkdir(exist_ok=True)

    if norm_mode == "minmax":
        im_norm = normalize_img(img_np)
    elif norm_mode == "percentile":
        im_norm = normalize_img_percentile(img_np, p_low=1.0, p_high=99.0)
    else:
        logger.warning(f"Unknown norm_mode={norm_mode!r}, falling back to percentile")
        im_norm = normalize_img_percentile(img_np, p_low=1.0, p_high=99.0)

    # Compute masks from the raw (pre-normalization) image so thresholds are in absolute units.
    img_raw = np.asarray(img_np)
    img_flat = img_raw.ravel()

    if img_flat.size == 0:
        logger.warning("Target image is empty; skipping threshold previews.")
    else:
        img_min_raw = float(np.min(img_flat))
        img_max_raw = float(np.max(img_flat))
        logger.info("Target image absolute range: min=%.8g max=%.8g", img_min_raw, img_max_raw)

        for thr in thresholds:
            threshold = float(thr)
            mask = img_raw > threshold
            actual_keep = float(mask.mean())

            fig, ax = plt.subplots(figsize=(8, 8))
            rgb = overlay_nonzero_red(im_norm, mask)
            ax.imshow(rgb)
            ax.set_title(
                "Target Image (b)\n"
                f"threshold > {threshold:.5e} | keep={actual_keep:.4%}"
            )
            ax.axis('off')

            out_name = f"target_thr_{threshold:.04f}.png".replace("+", "")
            plt.savefig(threshold_output_dir / out_name)
            plt.close(fig)

    logger.info(f"Saved thresholded images to {threshold_output_dir}")

    # --- 3. Volume Slices ---
    logger.info("Generating Volume Slices...")
    # vol shape is (X, Y, Z). We usually slice along Z (depth) or Y?
    # Context says Z is depth (index 2).
    # Let's plot grid of Z slices.
    
    nz = vol.shape[2]

    # Show more slices for better spatial context.
    # Keep it bounded so the figure doesn't become unreasonably large.
    # NOTE: 64 slices often isn't enough to see structure in large Z volumes.
    num_slices = int(min(100, int(nz)))
    indices = np.linspace(0, nz - 1, num_slices, dtype=int)

    # Uniform legend: use a single (vmin, vmax) for ALL slices.
    # Using global min/max is very sensitive to outliers; a robust global range
    # usually produces more readable plots while remaining consistent.
    global_vmin, global_vmax = robust_range_from_data(vol_np_sample, p_low=1.0, p_high=99.0)
    logger.info(
        "Volume slice colormap range (global, robust): vmin=%.6g vmax=%.6g",
        global_vmin,
        global_vmax,
    )

    ncols = int(np.ceil(np.sqrt(num_slices)))
    nrows = int(np.ceil(num_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.asarray(axes).ravel()

    shared_im = None
    
    for i, idx in enumerate(indices):
        # Slice: vol[:, :, idx] -> (X, Y).
        # imshow expects (H, W) -> (Y, X). So transpose.
        slice_data = vol[:, :, idx].numpy().T
        im = axes[i].imshow(slice_data, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[i].set_title(f"Z-Slice {idx}")
        # Keep axes visible for alignment debugging
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")

        if shared_im is None:
            shared_im = im

    # Hide any extra axes if grid > num_slices
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    # Single shared colorbar/legend for all slices (match visualize_density_slices.py style).
    if shared_im is not None:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(shared_im, cax=cbar_ax, label='Intensity')

    plt.savefig(output_dir / "volume_slices.png", bbox_inches='tight')
    plt.close()
    logger.info("Saved volume_slices.png")

    # --- 4. Video (Z-Scan) ---
    logger.info("Generating Volume Video (Z-scan)...")
    
    # Reduce resolution for video generation speed/size if needed
    # Volume might be huge (e.g. 1192x2048). 
    # Let's resize or just plot directly. Plotting 2Kx1K frames is slow.
    # Let's simple slice without resizing first, see performance.
    
    fig_vid, ax_vid = plt.subplots(figsize=(8, 8))

    # Pick frames: always scan the whole volume with a fixed stride.
    stride = 8
    frames = list(range(0, nz, stride))

    if not frames:
        logger.warning("No frames selected for video; skipping video export.")
        plt.close(fig_vid)
        return

    # Use a fixed robust range across all frames so the colorbar is meaningful.
    vid_vmin, vid_vmax = robust_range_from_data(vol_np_sample, p_low=1.0, p_high=99.0)
    logger.info("Video colormap range (robust): vmin=%.6g vmax=%.6g", vid_vmin, vid_vmax)

    # Initialize with the first selected slice
    first_slice = vol[:, :, frames[0]].numpy().T
    first_slice = _downsample_2d(first_slice, video_downsample)
    im_display = ax_vid.imshow(first_slice, cmap='viridis', animated=True, vmin=vid_vmin, vmax=vid_vmax)
    ax_vid.set_title(f"Z-Slice Video")
    ax_vid.axis('off')
    
    # Add colorbar
    fig_vid.colorbar(im_display, ax=ax_vid, fraction=0.046, pad=0.04, label='Intensity')

    def update(frame):
        # Frame is Z index
        slice_data = vol[:, :, frame].numpy().T
        slice_data = _downsample_2d(slice_data, video_downsample)
        im_display.set_data(slice_data)
        ax_vid.set_title(f"Z-Slice {frame}")
        return [im_display]

    logger.info(f"Video frames: {len(frames)} (nz={nz}, stride={stride}, downsample={video_downsample}x)")
    ani = FuncAnimation(fig_vid, update, frames=frames, blit=True)
    
    # Try saving as mp4 (requires ffmpeg)
    video_path = output_dir / "volume_scan.mp4"
    ani.save(video_path, writer='ffmpeg', fps=video_fps)
    logger.info(f"Saved video to {video_path}")

    plt.close(fig_vid)
    
    logger.info("Visualization Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs='?', default=None, help="Path to the .h5 or .pt file (Optional if using raw dirs)")
    parser.add_argument(
        "--norm",
        choices=["percentile", "minmax"],
        default="percentile",
        help="Normalization for visualization (percentile recommended for sparse images)",
    )
    parser.add_argument(
        "--video_downsample",
        type=int,
        default=2,
        help="Downsample each frame by striding (factor N reduces H and W by N)",
    )
    parser.add_argument("--video_fps", type=int, default=10, help="Frames per second for exported video")
    parser.add_argument(
        "--hist_decades",
        type=int,
        default=10,
        help="Number of decades for log histogram buckets (e.g., 6 -> 1e-6..1)",
    )
    
    # Raw Data Mode Arguments
    parser.add_argument("--input-dir", default="data/raw/lightsheet_vol_6.9", help="Raw volume directory (e.g. data/raw/lightsheet_vol_6.9)")
    parser.add_argument("--img-dir", default="data/raw/20um_imgs", help="Raw image directory (e.g. data/raw/20um_imgs)")
    parser.add_argument("--downsampling-rate", type=float, default=0.5, help="Downsampling rate for raw mode")
    parser.add_argument("--scale-factor", type=float, default=8.0, help="Scale factor for raw mode")

    args = parser.parse_args()
    
    # Logic: If raw dirs are provided, we use raw mode. file_path argument might be used to specify index (e.g. "pair_2")
    raw_mode = (args.input_dir is not None and args.img_dir is not None)
    
    if not raw_mode and not Path(args.file_path).exists():
        print(f"Error: File {args.file_path} does not exist and no raw directories provided.")
    else:
        visualize(
            args.file_path,
            args=args,
            norm_mode=args.norm,
            video_downsample=args.video_downsample,
            video_fps=args.video_fps,
            hist_log_bins=False,
            hist_decades=args.hist_decades,
        )