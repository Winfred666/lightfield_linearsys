import argparse
import torch
import glob
from pathlib import Path
import logging
import os
import sys
import time
import gc
import h5py
import numpy as np
import re
from typing import List, Tuple, Optional

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _fmt_time(dt: float) -> str:
    if dt < 1e-3:
        return f"{dt*1e6:.1f}us"
    if dt < 1:
        return f"{dt*1e3:.1f}ms"
    return f"{dt:.2f}s"


def _cuda_sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _log_cuda_mem(prefix: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        logger.info("%s | CUDA mem: alloc=%.1fMB reserved=%.1fMB peak=%.1fMB", prefix, alloc, reserv, peak)

def get_dimensions(data_dir):
    pattern = os.path.join(data_dir, "pair_*.h5")
    # Natural sort
    files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'pair_(\d+).h5', x).group(1)))
    
    if not files:
        raise FileNotFoundError(f"No pair_*.h5 files found in {data_dir}")
        
    with h5py.File(files[0], 'r') as f:
        # A shape: (X, Y, Z)
        A_shape = f['A'].shape
        X = A_shape[0]
        Y = A_shape[1]
        Z = A_shape[2]
        
    return X, Y, Z, files


def _build_pass_index(
    coords: torch.Tensor,
    batch_size: int,
    pass_start_batch: int,
    pass_end_batch: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]], List[torch.Tensor]]:
    """Build a contiguous list of coordinates for a pass."""
    total_points = coords.shape[0]
    pass_coords_list: List[torch.Tensor] = []
    slices: List[Tuple[int, int, int]] = []
    batch_coords: List[torch.Tensor] = []

    cursor = 0
    for b_idx in range(pass_start_batch, pass_end_batch):
        start_pt = b_idx * batch_size
        end_pt = min(start_pt + batch_size, total_points)
        b_coords = coords[start_pt:end_pt]
        batch_coords.append(b_coords)

        pass_coords_list.append(b_coords)
        next_cursor = cursor + b_coords.shape[0]
        slices.append((b_idx, cursor, next_cursor))
        cursor = next_cursor

    pass_coords = torch.cat(pass_coords_list, dim=0) if len(pass_coords_list) > 1 else pass_coords_list[0]
    return pass_coords, slices, batch_coords

def preprocess_points_optimized(
    data_dir: str = "data/processed",
    output_dir: str = "data/points",
    batch_size: int = 10000,
    batches_per_pass: int = 1,
    limit_batches: Optional[int] = None,
    save_dtype: str = "float32",
    log_every_pairs: int = 10,
    sync_timing: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    try:
        X_dim, Y_dim, Z_dim, files = get_dimensions(data_dir)
    except Exception as e:
        logger.error(str(e))
        return

    num_pairs = len(files)
    logger.info(f"Found {num_pairs} pair files. Dimensions: X={X_dim}, Y={Y_dim}, Z={Z_dim}")
    
    # Generate all coordinates
    xx = torch.arange(X_dim)
    yy = torch.arange(Y_dim)
    grid_x, grid_y = torch.meshgrid(xx, yy, indexing='ij')
    
    # Flatten
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1) # (N_points, 2)
    total_points = coords.shape[0]
    
    num_batches = (total_points + batch_size - 1) // batch_size
    
    if limit_batches is not None and limit_batches < num_batches:
        logger.info(f"Limiting to {limit_batches} batches out of {num_batches}.")
        num_batches = limit_batches
        
    logger.info(f"Total points: {total_points}. Batch size: {batch_size}. Total batches to process: {num_batches}")
    logger.info(f"Batches per pass: {batches_per_pass}")
    logger.info(
        "Implementation: per-pass spatial slicing from multiple HDF5 files."
    )
    
    # Outer loop: Passes
    num_passes = (num_batches + batches_per_pass - 1) // batches_per_pass
    
    for p_idx in range(num_passes):
        pass_start_batch = p_idx * batches_per_pass
        pass_end_batch = min(pass_start_batch + batches_per_pass, num_batches)
        
        current_pass_batches = pass_end_batch - pass_start_batch
        
        logger.info(f"=== Starting Pass {p_idx+1}/{num_passes} (Batches {pass_start_batch} to {pass_end_batch-1}) ===")
        t_pass_start = time.time()

        # Build one contiguous coord list for this pass.
        pass_coords, slices, batch_coords_list = _build_pass_index(
            coords=coords,
            batch_size=batch_size,
            pass_start_batch=pass_start_batch,
            pass_end_batch=pass_end_batch,
        )
        
        # Determine Bounding Box for Slicing
        pass_x = pass_coords[:, 0]
        pass_y = pass_coords[:, 1]
        min_x, max_x = pass_x.min().item(), pass_x.max().item()
        min_y, max_y = pass_y.min().item(), pass_y.max().item()
        
        # Pre-allocate CPU output per batch
        dtype = torch.float16 if save_dtype.lower() in {"fp16", "float16"} else torch.float32

        # Allocate pinned CPU staging buffers if using CUDA
        stage_pin = device.type == "cuda"
        A_stage = torch.empty((pass_coords.shape[0], Z_dim), dtype=dtype, pin_memory=stage_pin)
        b_stage = torch.empty((pass_coords.shape[0],), dtype=dtype, pin_memory=stage_pin)

        buffers = []
        for k in range(current_pass_batches):
            b_idx, s, e = slices[k]
            B_pts = e - s
            A_out = torch.empty((B_pts, num_pairs, Z_dim), dtype=dtype)
            b_out = torch.empty((B_pts, num_pairs), dtype=dtype)
            buffers.append(
                {
                    "idx": b_idx,
                    "coords": batch_coords_list[k],
                    "slice": (s, e),
                    "A": A_out,
                    "b": b_out,
                }
            )
            
        # Iterate over pair files
        for f_idx, f_path in enumerate(files):
            try:
                t_pair0 = time.time()
                
                with h5py.File(f_path, 'r') as f_h5:
                    # Read only the bounding box slice from HDF5
                    # A shape (X, Y, Z)
                    # b shape (Y, X) - Note this convention from preprocess_pair!
                    
                    A_ds = f_h5['A']
                    b_ds = f_h5['b']
                    
                    # Slicing
                    A_slice_np = A_ds[min_x:max_x+1, min_y:max_y+1, :]
                    b_slice_np = b_ds[min_y:max_y+1, min_x:max_x+1] # b is (Y, X)
                
                t_load = time.time()
                
                # Convert to tensor
                A_slice = torch.from_numpy(A_slice_np)
                b_slice = torch.from_numpy(b_slice_np)
                
                # Move to GPU for gathering
                if device.type == "cuda":
                    A_slice = A_slice.pin_memory()
                    b_slice = b_slice.pin_memory()
                    A_slice = A_slice.to(device, non_blocking=True)
                    b_slice = b_slice.to(device, non_blocking=True)
                    
                    # Indices for gathering relative to slice
                    # pass_x is global. local_x = pass_x - min_x
                    local_x = (pass_x - min_x).to(device, non_blocking=True)
                    local_y = (pass_y - min_y).to(device, non_blocking=True)
                else:
                    local_x = pass_x - min_x
                    local_y = pass_y - min_y

                t_h2d = time.time()
                
                # Gather
                # A_slice is (W_slice, H_slice, Z) -> (X, Y, Z)
                # local_x corresponds to dim 0 (X), local_y to dim 1 (Y)
                A_pass = A_slice[local_x, local_y, :]
                
                # b_slice is (H_slice, W_slice) -> (Y, X)
                # local_y corresponds to dim 0 (Y), local_x to dim 1 (X)
                b_pass = b_slice[local_y, local_x]

                t_gather = time.time()

                # D2H
                A_stage.copy_(A_pass.to(dtype), non_blocking=True)
                b_stage.copy_(b_pass.to(dtype), non_blocking=True)

                t_d2h = time.time()

                # Scatter to batches
                for k in range(current_pass_batches):
                    s, e = buffers[k]["slice"]
                    buffers[k]["A"][:, f_idx, :].copy_(A_stage[s:e])
                    buffers[k]["b"][:, f_idx].copy_(b_stage[s:e])

                t_done = time.time()

                if (f_idx % log_every_pairs == 0) or (f_idx == num_pairs - 1):
                    logger.info(
                        "Pass %d | pair %d/%d | load_slice=%s H2D=%s gather=%s D2H=%s scatter=%s total=%s",
                        p_idx + 1,
                        f_idx + 1,
                        num_pairs,
                        _fmt_time(t_load - t_pair0),
                        _fmt_time(t_h2d - t_load),
                        _fmt_time(t_gather - t_h2d),
                        _fmt_time(t_d2h - t_gather),
                        _fmt_time(t_done - t_d2h),
                        _fmt_time(t_done - t_pair0),
                    )
                
                del A_slice, b_slice, A_slice_np, b_slice_np
                
            except Exception as e:
                logger.error(f"Error processing pair {f_path}: {e}")
        
        # Save Batches
        logger.info("Saving batches for this pass...")
        t_save0 = time.time()
        for k in range(current_pass_batches):
            buf = buffers[k]
            b_idx = buf['idx']

            A_batch = buf["A"]
            b_batch = buf["b"]
            
            out_file = out_path / f"points_batch_{b_idx:04d}.pt"
            save_data = {
                'A': A_batch,
                'b': b_batch,
                'coords': buf['coords'],
                'batch_index': b_idx,
                'full_dims': (X_dim, Y_dim, Z_dim)
            }
            torch.save(save_data, out_file)
            logger.info(f"Saved {out_file} (A shape: {A_batch.shape}).")
            
            buffers[k] = None
            
        logger.info("Pass %d | saving time: %s", p_idx + 1, _fmt_time(time.time() - t_save0))
        logger.info(f"Pass {p_idx+1} completed in {time.time()-t_pass_start:.2f}s")
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed", help="Directory with pair_*.h5")
    # parser.add_argument("--output_dir", default="data/points", help="Output directory (default: derived from data_dir)")
    parser.add_argument("--batch_size", type=int, default=30000, help="Points per output file")
    parser.add_argument("--batches_per_pass", type=int, default=1, help="Number of batches to process in one pass (depends on RAM)")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches for testing")
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "fp16"],
        help="Dtype to store A/b on disk (float16 is smaller and faster to save).",
    )
    parser.add_argument(
        "--log_every_pairs",
        type=int,
        default=10,
        help="Log timing breakdown every N pair files.",
    )
    parser.add_argument(
        "--sync_timing",
        action="store_true",
        help="Synchronize CUDA for more accurate timings (slower).",
    )
    
    args = parser.parse_args()
    
    # Infer from data_dir
    args.output_dir = f"{args.data_dir.replace('processed', 'points')}_{args.batch_size}"

    logger.info(f"Data Dir: {args.data_dir}")
    logger.info(f"Output Dir: {args.output_dir}")
    
    preprocess_points_optimized(
        args.data_dir,
        args.output_dir,
        args.batch_size,
        args.batches_per_pass,
        args.limit_batches,
        save_dtype=args.save_dtype,
        log_every_pairs=args.log_every_pairs,
        sync_timing=args.sync_timing,
    )