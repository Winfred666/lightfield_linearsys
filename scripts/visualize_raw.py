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

# thresholds = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.9, 0.8, 0.7]
thresholds = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.9, 0.8, 0.7]


def setup_logging(output_dir):
    log_path = output_dir / "analysis.log"
    # Reset logging handlers to avoid mixing logs from different runs if called in same session
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
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
    file_paths,
    args=None, # Pass full args object for flexibility
    norm_mode="percentile",
    video_downsample: int = 1,
    video_fps: int = 10,
    hist_log_bins: bool = True,
    hist_decades: int = 6,
):
    global thresholds
    # Setup Output Directory
    if hasattr(args, 'output_dir') and args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"result/visualize_test/{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
        
    logger.info(f"Starting visualization for {len(file_paths)} files")
    logger.info(f"Output directory: {output_dir}")

    y_limit = getattr(args, 'y_limit', 1000)
    x_min = getattr(args, 'x_min', None)
    x_max = getattr(args, 'x_max', None)
    
    all_data = []

    # Load Data
    for file_path in file_paths:
        try:
            logger.info(f"Loading data from {file_path}...")
            
            # Check if raw mode arguments are provided
            raw_mode = False
            if (hasattr(args, 'input_dir') and args.input_dir and args.input_dir.lower() != "none" and 
                hasattr(args, 'img_dir') and args.img_dir and args.img_dir.lower() != "none"):
                raw_mode = True
                
            if raw_mode:
                 from LF_linearsys.io.preprocess_pair import preprocess_one_pair
                 from LF_linearsys.io.raw_pairs import find_raw_pairs
                 
                 pairs = find_raw_pairs(args.input_dir, args.img_dir)
                 target_pair = None
                 
                 # Try to match the specific file requested, or default to the first one
                 target_idx = 1
                 if file_path and "pair_" in str(file_path):
                     # Try to extract index from filename like pair_2.h5 or similar
                     import re
                     match = re.search(r'(\d+)', Path(file_path).name)
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
                     continue

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
                        # Try reconstruction or A for volume
                        vol = data.get('reconstruction')
                        if vol is None:
                            vol = data.get('A')
                        
                        # Try b for image
                        img = data.get('b')
                        
                        # If we only have volume, create dummy image
                        if vol is not None and img is None:
                            img = torch.zeros((1, 1))
                        # If we only have image, create dummy volume
                        elif img is not None and vol is None:
                            vol = torch.zeros((1, 1, 1))
                        # If we have neither as a dict, maybe it's the tensor itself?
                        elif vol is None and img is None:
                            # Fallback: assume the dict might contain the tensor under another key or just use the data if it was a tensor
                            vol = data
                            img = torch.zeros((1, 1))
                    elif isinstance(data, torch.Tensor):
                        vol = data
                        img = torch.zeros((1, 1))
                    else:
                        # Fallback if structure is different
                        try:
                            vol = data[0]
                            img = data[1]
                        except:
                            vol = data
                            img = torch.zeros((1, 1))
                
                # Final safety check
                if vol is None: vol = torch.zeros((1,1,1))
                if img is None: img = torch.zeros((1,1))
                
            logger.info(f"Volume Shape: {vol.shape}, Type: {vol.dtype}")
            logger.info(f"Image Shape: {img.shape}, Type: {img.dtype}")
            
            all_data.append({
                'vol': vol,
                'img': img,
                'label': Path(file_path).name
            })
            
        except Exception as e:
            logger.error(f"Failed to load file or process raw data: {e}", exc_info=True)
            continue

    if not all_data:
        logger.error("No data loaded. Exiting.")
        return

    # --- 1. Histograms ---
    logger.info("Generating Histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for i, data in enumerate(all_data):
        vol = data['vol']
        img = data['img']
        label = data['label']
        color = colors[i % len(colors)]
        
        # Convert to numpy
        img_np = img.numpy()
        vol_np = vol.numpy()
        
        # Log stats
        vol_flat = vol_np.ravel()
        if vol_flat.size > 0:
            vol_min = float(np.min(vol_flat))
            vol_max = float(np.max(vol_flat))
            vol_mean = float(np.mean(vol_flat))
            vol_median = float(np.median(vol_flat))
            logger.info(
                "[%s] Volume stats (A) [no subsample]: min=%.8g max=%.8g median=%.8g mean=%.8g",
                label, vol_min, vol_max, vol_median, vol_mean,
            )

        img_flat = img_np.ravel()
        if img_flat.size > 0:
            img_min = float(np.min(img_flat))
            img_max = float(np.max(img_flat))
            img_mean = float(np.mean(img_flat))
            img_median = float(np.median(img_flat))
            logger.info(
                "[%s] Image stats (pre-norm): min=%.8g max=%.8g median=%.8g mean=%.8g",
                label, img_min, img_max, img_median, img_mean
            )

        # Image Histogram
        if hist_log_bins:
            edges = log_hist_edges_from_data(img_flat, decades=hist_decades, include_zero=True)
            img_hist_vals = abs_values(img_flat)
            axes[0].hist(img_hist_vals, bins=edges, color=color, alpha=0.5, label=label)
            axes[0].set_xscale('log')
            if i == 0:
                axes[0].set_xlabel("|Intensity| (raw magnitude)")
                axes[0].set_ylabel("Count")
            if edges.size >= 3:
                axes[0].set_xlim(min(axes[0].get_xlim()[0], edges[1]), max(axes[0].get_xlim()[1], edges[-1]))
        else:
            # Linear histogram. 
            # If x_min and x_max are provided, we should probably use them for binning too to avoid wasting bins.
            hist_range = (x_min, x_max) if (x_min is not None and x_max is not None) else None
            axes[0].hist(img_flat, bins=100, range=hist_range, color=color, alpha=0.5, label=label)
            if i == 0:
                axes[0].set_xlabel("Intensity")
                axes[0].set_ylabel("Count")
        
        # Volume Histogram
        if hist_log_bins:
            edges = log_hist_edges_from_data(vol_flat, decades=hist_decades, include_zero=True)
            vol_hist_vals = abs_values(vol_flat)
            axes[1].hist(vol_hist_vals, bins=edges, color=color, alpha=0.5, label=label)
            axes[1].set_xscale('log')
            if i == 0:
                axes[1].set_xlabel("|Intensity| (raw magnitude)")
                axes[1].set_ylabel("Count")
            if edges.size >= 3:
                axes[1].set_xlim(min(axes[1].get_xlim()[0], edges[1]), max(axes[1].get_xlim()[1], edges[-1]))
        else:
            hist_range = (x_min, x_max) if (x_min is not None and x_max is not None) else None
            axes[1].hist(vol_flat, bins=100, range=hist_range, color=color, alpha=0.5, label=label)
            if i == 0:
                axes[1].set_xlabel("Intensity")
                axes[1].set_ylabel("Count")

    axes[0].set_title("Target Image (b) Histograms")
    axes[0].legend()
    axes[1].set_title("Volume (A) Histograms")
    axes[1].legend()
    
    if y_limit:
        axes[0].set_ylim(0, y_limit)
        axes[1].set_ylim(0, y_limit)
    
    if x_min is not None and x_max is not None:
        axes[0].set_xlim(x_min, x_max)
        axes[1].set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.savefig(output_dir / "histograms.png")
    plt.close()
    logger.info(f"Saved histograms.png (y-axis limit: {y_limit}, x-axis range: [{x_min}, {x_max}])")

    # --- For the rest of visualizations, use the first file only ---
    first_data = all_data[0]
    vol = first_data['vol']
    img = first_data['img']
    vol_np = vol.numpy()
    img_np = img.numpy()
    
    # --- 2. Threshold previews (absolute thresholds; hard-coded list) ---
    logger.info("Generating threshold previews for the first file...")
    threshold_output_dir = output_dir / "target_image_threshold_previews"
    threshold_output_dir.mkdir(exist_ok=True)

    if norm_mode == "minmax":
        im_norm = normalize_img(img_np)
    elif norm_mode == "percentile":
        im_norm = normalize_img_percentile(img_np, p_low=1.0, p_high=99.0)
    else:
        im_norm = normalize_img_percentile(img_np, p_low=1.0, p_high=99.0)

    img_raw = np.asarray(img_np)
    if img_raw.size > 0:
        for thr in thresholds:
            threshold = float(thr)
            mask = img_raw > threshold
            actual_keep = float(mask.mean())
            fig, ax = plt.subplots(figsize=(8, 8))
            rgb = overlay_nonzero_red(im_norm, mask)
            ax.imshow(rgb)
            ax.set_title(f"Target Image (b)\nthreshold > {threshold:.5e} | keep={actual_keep:.4%}")
            ax.axis('off')
            out_name = f"target_thr_{threshold:.04f}.png"
            plt.savefig(threshold_output_dir / out_name)
            plt.close(fig)

    # --- 3. Volume Slices (for first file) ---
    logger.info("Generating Volume Slices for the first file...")
    nz = vol.shape[2]
    num_slices = int(min(100, int(nz)))
    indices = np.linspace(0, nz - 1, num_slices, dtype=int)
    global_vmin, global_vmax = robust_range_from_data(vol_np, p_low=1.0, p_high=99.0)
    
    ncols = int(np.ceil(np.sqrt(num_slices)))
    nrows = int(np.ceil(num_slices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.asarray(axes).ravel()
    shared_im = None
    for i, idx in enumerate(indices):
        slice_data = vol[:, :, idx].numpy().T
        im = axes[i].imshow(slice_data, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[i].set_title(f"Z-Slice {idx}")
        axes[i].axis('off')
        if shared_im is None: shared_im = im
    for j in range(len(indices), len(axes)): axes[j].axis('off')
    if shared_im is not None:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(shared_im, cax=cbar_ax, label='Intensity')
    plt.savefig(output_dir / "volume_slices.png", bbox_inches='tight')
    plt.close()

    # --- 4. Video (Z-Scan) ---
    logger.info("Generating Volume Video (Z-scan) for the first file...")
    stride = 8
    frames = list(range(0, nz, stride))
    if frames:
        fig_vid, ax_vid = plt.subplots(figsize=(8, 8))
        vid_vmin, vid_vmax = robust_range_from_data(vol_np, p_low=1.0, p_high=99.0)
        first_slice = vol[:, :, frames[0]].numpy().T
        first_slice = _downsample_2d(first_slice, video_downsample)
        im_display = ax_vid.imshow(first_slice, cmap='viridis', animated=True, vmin=vid_vmin, vmax=vid_vmax)
        ax_vid.axis('off')
        fig_vid.colorbar(im_display, ax=ax_vid, fraction=0.046, pad=0.04, label='Intensity')
        def update(frame):
            slice_data = vol[:, :, frame].numpy().T
            slice_data = _downsample_2d(slice_data, video_downsample)
            im_display.set_data(slice_data)
            ax_vid.set_title(f"Z-Slice {frame}")
            return [im_display]
        ani = FuncAnimation(fig_vid, update, frames=frames, blit=True)
        ani.save(output_dir / "volume_scan.mp4", writer='ffmpeg', fps=video_fps)
        plt.close(fig_vid)
    
    logger.info("Visualization Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", nargs='*', default=[], help="Path(s) to the .h5 or .pt file(s)")
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
    parser.add_argument("--output_dir", help="Explicit output directory")
    parser.add_argument("--y_limit", type=int, default=1000, help="Y-axis limit for histograms")
    parser.add_argument("--x_min", type=float, help="X-axis minimum for histograms")
    parser.add_argument("--x_max", type=float, help="X-axis maximum for histograms")
    
    # Raw Data Mode Arguments
    parser.add_argument("--input-dir", default=None, help="Raw volume directory")
    parser.add_argument("--img-dir", default=None, help="Raw image directory")
    parser.add_argument("--downsampling-rate", type=float, default=0.5, help="Downsampling rate for raw mode")
    parser.add_argument("--scale-factor", type=float, default=8.0, help="Scale factor for raw mode")

    args = parser.parse_args()
    
    raw_mode = (args.input_dir is not None and args.input_dir.lower() != "none" and 
                args.img_dir is not None and args.img_dir.lower() != "none")
    
    if not raw_mode and not args.file_paths:
        print("Error: No files specified and no raw directories provided.")
    else:
        visualize(
            args.file_paths,
            args=args,
            norm_mode=args.norm,
            video_downsample=args.video_downsample,
            video_fps=args.video_fps,
            hist_log_bins=False,
            hist_decades=args.hist_decades,
        )
