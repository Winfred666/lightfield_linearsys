import argparse
import glob
import gc
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch

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



def get_dimensions(data_dir: str) -> tuple[int, int, int, list[str]]:
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


def save_points_batch(out_file: Path, *, A: torch.Tensor, b: torch.Tensor, coords: torch.Tensor, batch_index: int, full_dims: tuple[int, int, int]) -> None:
    """Save a packed points batch to disk (torch .pt).

    Expected shapes:
      A: (B_pts, num_pairs, Z)
      b: (B_pts, num_pairs)
      coords: (B_pts, 2) with (x,y)
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'A': A,
            'b': b,
            'coords': coords,
            'batch_index': int(batch_index),
            'full_dims': tuple(full_dims),
        },
        out_file,
    )


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


def pack_points_from_pair_h5(
    *,
    data_dir: str,
    output_dir: str,
    batch_size: int = 10000,
    batches_per_pass: int = 1,
    limit_batches: Optional[int] = None,
    save_dtype: str = "float32",
    log_every_pairs: int = 10,
    save_batches: bool = True,
) -> list[Path]:
    """Pack per-pixel equations from multiple pair_*.h5 into point batches.

    This is the original behavior of preprocess_points_optimized(), kept as a
    function so it can be reused and unit-tested.

    Returns a list of written batch .pt files (empty if save_batches=False).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        X_dim, Y_dim, Z_dim, files = get_dimensions(data_dir)
    except Exception as e:
        logger.error(str(e))
        return []

    num_pairs = len(files)
    logger.info("Found %d pair files. Dimensions: X=%d, Y=%d, Z=%d", num_pairs, X_dim, Y_dim, Z_dim)

    # Generate all coordinates
    xx = torch.arange(X_dim)
    yy = torch.arange(Y_dim)
    grid_x, grid_y = torch.meshgrid(xx, yy, indexing='ij')

    # Flatten
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N_points, 2)
    total_points = coords.shape[0]

    num_batches = (total_points + batch_size - 1) // batch_size
    if limit_batches is not None and limit_batches < num_batches:
        logger.info("Limiting to %d batches out of %d.", limit_batches, num_batches)
        num_batches = limit_batches

    logger.info("Total points: %d. Batch size: %d. Total batches to process: %d", total_points, batch_size, num_batches)
    logger.info("Batches per pass: %d", batches_per_pass)

    # Outer loop: Passes
    num_passes = (num_batches + batches_per_pass - 1) // batches_per_pass
    written: list[Path] = []

    for p_idx in range(num_passes):
        pass_start_batch = p_idx * batches_per_pass
        pass_end_batch = min(pass_start_batch + batches_per_pass, num_batches)

        current_pass_batches = pass_end_batch - pass_start_batch

        logger.info(
            "=== Starting Pass %d/%d (Batches %d to %d) ===",
            p_idx + 1,
            num_passes,
            pass_start_batch,
            pass_end_batch - 1,
        )

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

        dtype = torch.float16 if save_dtype.lower() in {"fp16", "float16"} else torch.float32
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
                    A_ds = f_h5['A']
                    b_ds = f_h5['b']

                    A_slice_np = A_ds[min_x:max_x + 1, min_y:max_y + 1, :]
                    b_slice_np = b_ds[min_y:max_y + 1, min_x:max_x + 1]  # b is (Y, X)

                t_load = time.time()

                A_slice = torch.from_numpy(A_slice_np)
                b_slice = torch.from_numpy(b_slice_np)

                if device.type == "cuda":
                    A_slice = A_slice.pin_memory().to(device, non_blocking=True)
                    b_slice = b_slice.pin_memory().to(device, non_blocking=True)
                    local_x = (pass_x - min_x).to(device, non_blocking=True)
                    local_y = (pass_y - min_y).to(device, non_blocking=True)
                else:
                    local_x = pass_x - min_x
                    local_y = pass_y - min_y

                t_h2d = time.time()

                A_pass = A_slice[local_x, local_y, :]
                b_pass = b_slice[local_y, local_x]

                t_gather = time.time()

                A_stage.copy_(A_pass.to(dtype), non_blocking=True)
                b_stage.copy_(b_pass.to(dtype), non_blocking=True)

                t_d2h = time.time()

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
                logger.error("Error processing pair %s: %s", f_path, e)

        # Save Batches
        for k in range(current_pass_batches):
            buf = buffers[k]
            b_idx = buf['idx']
            if save_batches:
                out_file = out_path / f"points_batch_{b_idx:04d}.pt"
                save_points_batch(
                    out_file,
                    A=buf["A"],
                    b=buf["b"],
                    coords=buf['coords'],
                    batch_index=b_idx,
                    full_dims=(X_dim, Y_dim, Z_dim),
                )
                written.append(out_file)
                logger.info("Saved %s (A shape: %s).", out_file, tuple(buf["A"].shape))

            buffers[k] = None

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return written


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
    # Backward-compatible wrapper.
    _ = sync_timing
    pack_points_from_pair_h5(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        batches_per_pass=batches_per_pass,
        limit_batches=limit_batches,
        save_dtype=save_dtype,
        log_every_pairs=log_every_pairs,
        save_batches=True,
    )


def preprocess_points_from_raw(
    *,
    input_dir: str,
    img_dir: str,
    output_dir: str,
    downsampling_rate: float,
    scale_factor: float = 8.0,
    max_pairs: Optional[int] = None,
    batch_size: int = 10000,
    batches_per_pass: int = 1,
    limit_batches: Optional[int] = None,
    save_dtype: str = "float32",
    log_every_pairs: int = 10,
) -> list[Path]:
    """Build points batches directly from raw (vol,image) pairs.

    This mirrors preprocess_pair's dataset format, but skips writing pair_*.h5.
    It calls `preprocess_one_pair` per index to produce (A_m,b_m) tensors in
    memory, then packs them into point batches.

    Notes:
      - Intended for small/moderate number of pairs. For very large datasets,
        intermediate HDF5 pair files may still be preferable.
    """
    from LF_linearsys.io.preprocess_pair import preprocess_one_pair
    from LF_linearsys.io.raw_pairs import find_raw_pairs

    pairs = find_raw_pairs(input_dir=input_dir, img_dir=img_dir, max_pairs=max_pairs)
    if not pairs:
        raise FileNotFoundError(f"No matching raw pairs found in {input_dir} and {img_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Found %d raw pairs to process.", len(pairs))

    # Determine dimensions from the first pair only.
    # (We expect all pairs to share the same (X,Y,Z) after preprocessing.)
    t_dim0 = time.time()
    A0_cpu, b0_cpu = preprocess_one_pair(
        vol_path=pairs[0].vol_path,
        img_path=pairs[0].img_path,
        downsampling_rate=downsampling_rate,
        scale_factor=scale_factor,
        device=device,
    )
    X_dim, Y_dim, Z_dim = A0_cpu.shape
    num_pairs = len(pairs)
    logger.info(
        "Raw-mode dimensions: X=%d Y=%d Z=%d (from idx=%d) in %s",
        X_dim,
        Y_dim,
        Z_dim,
        pairs[0].idx,
        _fmt_time(time.time() - t_dim0),
    )

    # Generate all coordinates
    xx = torch.arange(X_dim)
    yy = torch.arange(Y_dim)
    grid_x, grid_y = torch.meshgrid(xx, yy, indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    total_points = coords.shape[0]

    num_batches = (total_points + batch_size - 1) // batch_size
    if limit_batches is not None and limit_batches < num_batches:
        num_batches = limit_batches
    num_passes = (num_batches + batches_per_pass - 1) // batches_per_pass

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if save_dtype.lower() in {"fp16", "float16"} else torch.float32
    written: list[Path] = []

    for p_idx in range(num_passes):
        pass_start_batch = p_idx * batches_per_pass
        pass_end_batch = min(pass_start_batch + batches_per_pass, num_batches)
        current_pass_batches = pass_end_batch - pass_start_batch

        pass_coords, slices, batch_coords_list = _build_pass_index(
            coords=coords,
            batch_size=batch_size,
            pass_start_batch=pass_start_batch,
            pass_end_batch=pass_end_batch,
        )

        pass_x = pass_coords[:, 0].long()
        pass_y = pass_coords[:, 1].long()

        buffers = []
        for k in range(current_pass_batches):
            b_idx, s, e = slices[k]
            B_pts = e - s
            A_out = torch.empty((B_pts, num_pairs, Z_dim), dtype=dtype)
            b_out = torch.empty((B_pts, num_pairs), dtype=dtype)
            buffers.append(
                {"idx": b_idx, "coords": batch_coords_list[k], "slice": (s, e), "A": A_out, "b": b_out}
            )

        # Stream over pairs (do NOT keep all A/b in memory).
        # This is analogous to pack_points_from_pair_h5() but with preprocess_one_pair() as the reader.
        for m, pair in enumerate(pairs):
            t_pair0 = time.time()
            A_cpu, b_cpu = preprocess_one_pair(
                vol_path=pair.vol_path,
                img_path=pair.img_path,
                downsampling_rate=downsampling_rate,
                scale_factor=scale_factor,
                device=device,
            )

            if A_cpu.shape != (X_dim, Y_dim, Z_dim):
                raise ValueError(
                    f"Inconsistent A shape for idx={pair.idx}: got {tuple(A_cpu.shape)} expected {(X_dim, Y_dim, Z_dim)}"
                )

            # gather for pass
            A_pass = A_cpu[pass_x, pass_y, :].to(dtype)
            b_pass = b_cpu[pass_y, pass_x].to(dtype)

            for k in range(current_pass_batches):
                s, e = buffers[k]["slice"]
                buffers[k]["A"][:, m, :].copy_(A_pass[s:e])
                buffers[k]["b"][:, m].copy_(b_pass[s:e])

            if (m % log_every_pairs == 0) or (m == num_pairs - 1):
                logger.info(
                    "Pass %d/%d | pair %d/%d (idx=%d) | total=%s",
                    p_idx + 1,
                    num_passes,
                    m + 1,
                    num_pairs,
                    pair.idx,
                    _fmt_time(time.time() - t_pair0),
                )

            del A_cpu, b_cpu, A_pass, b_pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for k in range(current_pass_batches):
            buf = buffers[k]
            out_file = out_path / f"points_batch_{buf['idx']:04d}.pt"
            save_points_batch(
                out_file,
                A=buf["A"],
                b=buf["b"],
                coords=buf["coords"],
                batch_index=buf["idx"],
                full_dims=(X_dim, Y_dim, Z_dim),
            )
            written.append(out_file)
            logger.info("Saved %s (A shape: %s).", out_file, tuple(buf["A"].shape))

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mode A: read existing processed pair_*.h5
    parser.add_argument("--data_dir", default=None, help="Directory with pair_*.h5 (processed pairs)")
    # Mode B: read raw volume/image pairs and call preprocess_one_pair
    parser.add_argument("--input_dir", default=None, help="Directory containing Interp_Vol_ID_*.pt (raw volumes)")
    parser.add_argument("--img_dir", default=None, help="Directory containing 1scan (idx).tif (raw images)")
    parser.add_argument("--downsampling_rate", type=float, default=None, help="Downsampling rate for raw-mode (required if --input_dir/--img_dir used)")
    parser.add_argument("--scale_factor", type=float, default=8.0, help="Scale factor for raw-mode (volume scaling uses scale_factor * downsampling_rate)")

    parser.add_argument("--output_dir", default=None, help="Output directory for points_batch_*.pt")
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

    # Determine mode
    using_pair_h5 = args.data_dir is not None
    using_raw = (args.input_dir is not None) or (args.img_dir is not None)

    if using_pair_h5 and using_raw:
        raise ValueError("Choose either --data_dir (pair_*.h5) OR (--input_dir and --img_dir) raw mode, not both.")

    if not using_pair_h5 and not using_raw:
        raise ValueError("You must provide either --data_dir (pair_*.h5) or --input_dir/--img_dir (raw mode).")

    if args.output_dir is None:
        if using_pair_h5:
            args.output_dir = f"{str(args.data_dir).replace('processed', 'points')}_{args.batch_size}"
        else:
            args.output_dir = f"data/points_raw_{args.batch_size}"

    logger.info("Output Dir: %s", args.output_dir)

    if using_pair_h5:
        logger.info("Pair-H5 mode. Data Dir: %s", args.data_dir)
        preprocess_points_optimized(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            batches_per_pass=args.batches_per_pass,
            limit_batches=args.limit_batches,
            save_dtype=args.save_dtype,
            log_every_pairs=args.log_every_pairs,
            sync_timing=args.sync_timing,
        )
    else:
        if args.input_dir is None or args.img_dir is None or args.downsampling_rate is None:
            raise ValueError("Raw mode requires --input_dir, --img_dir, and --downsampling_rate")
        logger.info("Raw mode. input_dir=%s img_dir=%s downsampling_rate=%s", args.input_dir, args.img_dir, args.downsampling_rate)
        preprocess_points_from_raw(
            input_dir=args.input_dir,
            img_dir=args.img_dir,
            output_dir=args.output_dir,
            downsampling_rate=float(args.downsampling_rate),
            scale_factor=float(args.scale_factor),
            batch_size=args.batch_size,
            batches_per_pass=args.batches_per_pass,
            limit_batches=args.limit_batches,
            save_dtype=args.save_dtype,
            log_every_pairs=args.log_every_pairs,
        )