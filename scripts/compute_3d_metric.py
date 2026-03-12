import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import numpy as np
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import shutil
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from LF_linearsys.utils.visualize_slices import _ensure_cpu_float

def visualize_side_projection(recon, gt, out_path):
    recon = _ensure_cpu_float(recon)
    gt = _ensure_cpu_float(gt)
    
    # Ensure shapes match (truncate to smaller Z if needed)
    min_z = min(recon.shape[2], gt.shape[2])
    recon = recon[:, :, :min_z]
    gt = gt[:, :, :min_z]

    r_proj = np.sum(recon.numpy(), axis=0)  # (Y,Z)
    g_proj = np.sum(gt.numpy(), axis=0)     # (Y,Z)

    diff_proj = r_proj - g_proj
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Global vmin/vmax for consistent comparison
    vmin = min(r_proj.min(), g_proj.min())
    vmax = max(r_proj.max(), g_proj.max())
    
    # Recon
    im0 = axes[0].imshow(r_proj, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Recon Projection (Side)")
    axes[0].axis("off")
    
    # GT
    im1 = axes[1].imshow(g_proj, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("GT Projection (Side)")
    axes[1].axis("off")
    
    # Shared Colorbar for Recon/GT
    # Add colorbar to the first two plots (shared)
    # We can add it to one of the axes or create a separate axis.
    # A simple way is to add it to axes[1] or using a list of axes.
    fig.colorbar(im0, ax=[axes[0], axes[1]], fraction=0.046, pad=0.04, format='%.2e')
    
    # Error
    err_max = max(abs(diff_proj.min()), abs(diff_proj.max())) + 1e-6
    im2 = axes[2].imshow(diff_proj, cmap="coolwarm", vmin=-err_max, vmax=err_max, origin="lower")
    axes[2].set_title("Difference (Recon - GT)")
    axes[2].axis("off")
    
    # Separate Colorbar for Error
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, format='%.2e')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def visualize_comparison_video(recon, gt, out_path, fps=10):
    recon = _ensure_cpu_float(recon)
    gt = _ensure_cpu_float(gt)
    
    # Ensure shapes match (truncate to smaller Z if needed)
    min_z = min(recon.shape[2], gt.shape[2])
    recon = recon[:, :, :min_z]
    gt = gt[:, :, :min_z]
    
    recon_np = recon.numpy()
    gt_np = gt.numpy()
    
    # Determine global vmin/vmax for consistent visualization
    # Subsample for speed if volume is huge
    subsample = recon_np[::2, ::2, ::2] 
    if subsample.size > 0:
        vmin = float(np.percentile(subsample, 0))
        vmax = float(np.percentile(subsample, 100))
        if vmax <= vmin:
            vmin, vmax = float(np.min(subsample)), float(np.max(subsample))
    else:
        vmin, vmax = 0.0, 1.0
        
    tmp_dir = Path(tempfile.mkdtemp(dir=out_path.parent))
    frames = []
    
    print(f"Generating video frames for {out_path.name}...")
    
    try:
        for z in range(min_z):
            r_slice = recon_np[:, :, z].T # Transpose for (Y, X) image convention
            g_slice = gt_np[:, :, z].T
            diff_slice = r_slice - g_slice
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Recon
            im0 = axes[0].imshow(r_slice, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            axes[0].set_title(f"Reconstruction (z={z})")
            axes[0].axis("off")
            
            # GT
            im1 = axes[1].imshow(g_slice, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            axes[1].set_title(f"Ground Truth (z={z})")
            axes[1].axis("off")
            
            # Error
            # Use max abs error for symmetric colorbar
            err_max = max(abs(np.min(diff_slice)), abs(np.max(diff_slice))) + 1e-6
            im2 = axes[2].imshow(diff_slice, cmap="coolwarm", vmin=-err_max, vmax=err_max, origin="lower")
            axes[2].set_title("Difference (Recon - GT)")
            axes[2].axis("off")
            
            # Colorbars
            # Shared for Recon/GT
            # fig.colorbar(im0, ax=[axes[0], axes[1]], fraction=0.046, pad=0.04) 
            # Separate for Error
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, format='%.2e')
            
            plt.tight_layout()
            frame_path = tmp_dir / f"frame_{z:04d}.png"
            plt.savefig(frame_path, dpi=100)
            plt.close(fig)
            frames.append(frame_path)
            
        # Write video
        print(f"Encoding video {out_path}...")
        with imageio.get_writer(out_path, fps=fps, format="FFMPEG") as writer:
            for frame_path in frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)
                
    finally:
        # Cleanup
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

def compute_metrics(recon_path, gt_path, data_range_mode="gt_peak"):
    print(f"Loading reconstruction from {recon_path}...")
    try:
        recon_data = torch.load(recon_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {recon_path}: {e}")
        return

    if isinstance(recon_data, dict) and 'reconstruction' in recon_data:
        recon = recon_data['reconstruction']
    else:
        recon = recon_data
    
    print(f"Loading GT from {gt_path}...")
    try:
        gt = torch.load(gt_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {gt_path}: {e}")
        return

    # Ensure shapes match
    if recon.shape != gt.shape:
        print(f"Shape mismatch: Recon {recon.shape} vs GT {gt.shape}. Truncating to common size.")
        min_x = min(recon.shape[0], gt.shape[0])
        min_y = min(recon.shape[1], gt.shape[1])
        min_z = min(recon.shape[2], gt.shape[2])
        recon = recon[:min_x, :min_y, :min_z]
        gt = gt[:min_x, :min_y, :min_z]

    # PARTIAL AREA: crop out the front z-slices
    # front_z_slice = 5
    # recon = recon[:, :, front_z_slice:]
    # gt = gt[:, :, front_z_slice:]

    # Convert to float
    recon = recon.float()
    gt = gt.float()
    
    # MSE
    mse = F.mse_loss(recon, gt).item()
    
    # PSNR
    # Dynamic range logic
    if isinstance(data_range_mode, (float, int)):
        data_range = float(data_range_mode)
    elif data_range_mode == "gt_peak":
        data_range = gt.max().item()
    elif data_range_mode == "gt_range":
        data_range = (gt.max() - gt.min()).item()
    else:
        # Fallback
        data_range = 1.0
        
    if data_range <= 0: data_range = 1.0
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
        
    print("-" * 30)
    print(f"Comparison Result:")
    print(f"Reconstruction: {recon_path}")
    print(f"Ground Truth:   {gt_path}")
    print(f"MSE:            {mse:.6e}")
    print(f"PSNR:           {psnr:.2f} dB")
    print(f"PSNR data_range (MAX): {data_range} (mode={data_range_mode})")
    print("-" * 30)

    # Output directory
    recon_p = Path(recon_path)
    gt_p = Path(gt_path)
    # Avoid collision if gt name is same (reconstruction.pt)
    # Use parent folder name of GT if filename is generic
    gt_name_tag = gt_p.parent.name if gt_p.name == "reconstruction.pt" else gt_p.stem
    
    output_dir = recon_p.parent / f"compare_{gt_name_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Metrics
    with open(output_dir / "metrics.txt", "w") as f:
            f.write(f"Reconstruction: {recon_path}\n")
            f.write(f"Ground Truth: {gt_path}\n")
            f.write(f"MSE: {mse:.6e}\n")
            f.write(f"PSNR: {psnr:.2f} dB\n")
            f.write(f"Data Range: {data_range}\n")
            
    # Visualize Video
    viz_path = output_dir / "comparison_video.mp4"
    visualize_comparison_video(recon, gt, viz_path)
    
    # Visualize Side Projection
    side_proj_path = output_dir / "side_projection.png"
    visualize_side_projection(recon, gt, side_proj_path)
    
    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recon_path", help="Path to reconstruction.pt")
    parser.add_argument("--gt-path", default="data/synthetic/bunny/gt_volume.pt")
    parser.add_argument(
        "--data-range",
        default="gt_peak",
        help="PSNR data range (MAX). Use: gt_peak | gt_range | 1 | <float>.",
    )
    args = parser.parse_args()

    # allow numeric values
    try:
        dr = float(args.data_range)
        data_range_mode = dr
    except ValueError:
        data_range_mode = args.data_range

    compute_metrics(args.recon_path, args.gt_path, data_range_mode=data_range_mode)