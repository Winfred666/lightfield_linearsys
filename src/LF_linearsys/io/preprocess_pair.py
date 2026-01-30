import torch
from pathlib import Path
import sys
import os
import h5py
import numpy as np
import torch.nn.functional as F
import argparse

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from LF_linearsys.io.readers import read_volume, read_image, scale_volume
from LF_linearsys.io.raw_pairs import find_raw_pairs
import time


def _parse_int_tuple(s: str, *, n: int, name: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.replace("(", "").replace(")", "").split(",") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(f"{name} must have {n} integers, got {len(parts)} from: {s!r}")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"{name} must be integers, got: {s!r}") from e


def _validate_and_normalize_crop_boxes(
    *,
    crop_box_b: tuple[int, int, int, int] | None,
    crop_box_A: tuple[int, int, int, int, int, int] | None,
    img_shape_yx: tuple[int, int],
    vol_shape_xyz: tuple[int, int, int],
    scale_factor: float,
) -> tuple[
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int, int, int] | None,
]:
    """Validate crop boxes and ensure A/B crop sizes are consistent.

    Conventions:
      - b is (Y, X). crop_box_b is (x_min, y_min, x_max, y_max) in that image space.
      - A is (X, Y, Z). crop_box_A is (x_min, y_min, z_min, x_max, y_max, z_max).

    Requirement:
      - The crop boxes are specified in the *current* spaces before any new
        processing in this function.
      - They must already account for scaling ratios such that:
          (x_max-x_min)_b == round((x_max-x_min)_A * scale_factor)
          (y_max-y_min)_b == round((y_max-y_min)_A * scale_factor)
    """
    if crop_box_b is None and crop_box_A is None:
        return None, None

    if crop_box_b is None or crop_box_A is None:
        raise ValueError("Must provide both crop_box_b and crop_box_A, or neither.")

    x0b, y0b, x1b, y1b = crop_box_b
    x0a, y0a, z0a, x1a, y1a, z1a = crop_box_A

    if x0b < 0 or y0b < 0 or x1b <= x0b or y1b <= y0b:
        raise ValueError(f"Invalid crop_box_b={crop_box_b}. Expected (x0,y0,x1,y1) with x1>x0,y1>y0 and >=0")
    if x0a < 0 or y0a < 0 or z0a < 0 or x1a <= x0a or y1a <= y0a or z1a <= z0a:
        raise ValueError(
            f"Invalid crop_box_A={crop_box_A}. Expected (x0,y0,z0,x1,y1,z1) with max>min and >=0"
        )

    Y_i, X_i = img_shape_yx
    X_v, Y_v, Z_v = vol_shape_xyz
    if x1b > X_i or y1b > Y_i:
        raise ValueError(f"crop_box_b={crop_box_b} out of bounds for image shape (Y,X)={img_shape_yx}")
    if x1a > X_v or y1a > Y_v or z1a > Z_v:
        raise ValueError(f"crop_box_A={crop_box_A} out of bounds for volume shape (X,Y,Z)={vol_shape_xyz}")

    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor}")

    dx_a = x1a - x0a
    dy_a = y1a - y0a
    dx_b = x1b - x0b
    dy_b = y1b - y0b

    exp_dx_b = int(round(dx_a * scale_factor))
    exp_dy_b = int(round(dy_a * scale_factor))
    if dx_b != exp_dx_b or dy_b != exp_dy_b:
        raise ValueError(
            "crop boxes mismatch scaling ratio. "
            f"Got crop_box_A size (dx,dy)=({dx_a},{dy_a}), scale_factor={scale_factor}. "
            f"Expected crop_box_b size (dx,dy)=({exp_dx_b},{exp_dy_b}) but got ({dx_b},{dy_b})."
        )

    return crop_box_b, crop_box_A


def save_pair_h5(h5_path: Path, A_cpu: torch.Tensor, b_cpu: torch.Tensor) -> None:
    """Save a processed (A,b) pair to an HDF5 file.

    Args:
        h5_path: Destination file.
        A_cpu: (X, Y, Z) tensor on CPU.
        b_cpu: (Y, X) tensor on CPU.

    Notes:
        - Keeps the original chunking strategy used by this repo.
        - Stores as float32.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Chunking strategies:
    # A: (X, Y, Z). We often read sub-regions in X,Y. Z is relatively small.
    chunks_A = (32, 32, int(A_cpu.shape[2]))
    chunks_b = (32, 32)

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('A', data=A_cpu.numpy(), chunks=chunks_A, dtype='f4')
        f.create_dataset('b', data=b_cpu.numpy(), chunks=chunks_b, dtype='f4')


def preprocess_one_pair(
    *,
    vol_path: Path,
    img_path: Path,
    downsampling_rate: float,
    scale_factor: float = 8.0,
    crop_box_b: tuple[int, int, int, int] | None = None,
    crop_box_A: tuple[int, int, int, int, int, int] | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read+process a single (volume,image) pair into (A,b) tensors.

    Returns:
        (A_cpu, b_cpu)
          - A_cpu: (X, Y, Z) float32 on CPU
          - b_cpu: (Y, X) float32 on CPU

    This function encapsulates all the heavy preprocessing previously embedded
    inside preprocess_dataset().
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read
    vol = read_volume(vol_path)  # (X, Y, Z)
    img = read_image(img_path)   # (Y, X) numpy
    img = torch.from_numpy(img).float()

    # Validate + apply manual crop *before* any new processing.
    # User provided crop boxes are expected to already consider scale ratios.
    effective_vol_scale = scale_factor * downsampling_rate
    crop_box_b, crop_box_A = _validate_and_normalize_crop_boxes(
        crop_box_b=crop_box_b,
        crop_box_A=crop_box_A,
        img_shape_yx=tuple(img.shape),
        vol_shape_xyz=tuple(vol.shape),
        scale_factor=scale_factor,
    )
    if crop_box_b is not None and crop_box_A is not None:
        x0b, y0b, x1b, y1b = crop_box_b
        x0a, y0a, z0a, x1a, y1a, z1a = crop_box_A
        # img is (Y, X)
        img = img[y0b:y1b, x0b:x1b]
        # vol is (X, Y, Z)
        vol = vol[x0a:x1a, y0a:y1a, z0a:z1a]

    # Move to device
    vol = vol.to(device)
    img = img.to(device)

    # Downsample Image (b)
    if downsampling_rate != 1.0:
        # img is (Y, X). Need (1, 1, Y, X) for interpolate
        img_in = img.unsqueeze(0).unsqueeze(0)
        img_down = F.interpolate(img_in, scale_factor=downsampling_rate, mode='bilinear', align_corners=False)
        img = img_down.squeeze(0).squeeze(0)

    # Calculate effective scale for Volume (A)
    # We want to match the downsampled image space.
    # NOTE: effective_vol_scale already computed above for optional crop validation.

    # Optimization: Crop Volume BEFORE scaling to save memory.
    X_v_orig, Y_v_orig, Z_v_orig = vol.shape
    Y_i_curr, X_i_curr = img.shape

    # Calculate target crop size in Low Res (Volume) domain for Y
    # We target the downsampled image Y size.
    # Image Y ~= Original Volume Y * effective_vol_scale.
    target_Y_highres = min(Y_i_curr, int(Y_v_orig * effective_vol_scale))

    if effective_vol_scale > 0:
        target_Y_lowres = int(target_Y_highres / effective_vol_scale)
    else:
        target_Y_lowres = Y_v_orig

    if target_Y_lowres < Y_v_orig:
        start_y_v = (Y_v_orig - target_Y_lowres) // 2
        vol = vol[:, start_y_v: start_y_v + target_Y_lowres, :]

    # Chunked Scaling on GPU to avoid OOM
    Z_v_curr = vol.shape[2]
    chunk_size = 25
    num_chunks = (Z_v_curr + chunk_size - 1) // chunk_size
    
    Z_chunks = []
    with torch.no_grad():
        for i in range(num_chunks):
            z_start_in = i * chunk_size
            
            # Ensure overlap for interpolation if not the last chunk
            if i < num_chunks - 1:
                z_end_in = min((i + 1) * chunk_size + 1, Z_v_curr)
            else:
                z_end_in = Z_v_curr
            
            # If slice is empty or invalid, skip
            if z_start_in >= z_end_in:
                continue

            vol_chunk = vol[:, :, z_start_in: z_end_in]
            
            # Calculate local z target size for this chunk
            # The chunk covers input range [z_start_in, z_end_in)
            # The output should cover [z_start_in * scale, z_end_target * scale)
            # But wait, purely scaling a chunk independently requires care with boundary alignment.
            # The original code just appended results.
            
            vol_chunk_scaled = scale_volume(vol_chunk, scale_factor=effective_vol_scale)

            # Crop output to target Z size corresponding to the *valid* part of this chunk
            # Valid input size is chunk_size (except last one).
            # Output size should be valid_input_size * scale.
            
            valid_input_z = min((i + 1) * chunk_size, Z_v_curr) - z_start_in
            target_z_chunk = int(valid_input_z * effective_vol_scale)
            
            vol_chunk_scaled = vol_chunk_scaled[:, :, :target_z_chunk]

            # Move to CPU immediately
            Z_chunks.append(vol_chunk_scaled.cpu())

            del vol_chunk
            del vol_chunk_scaled
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Concatenate on CPU
    vol_scaled = torch.cat(Z_chunks, dim=2)  # On CPU

    # Now dimensions
    X_v, Y_v, Z_v = vol_scaled.shape

    # Target Crop Size for X
    target_X = min(X_v, X_i_curr)

    # Crop Image in X (width) to match Vol
    if X_i_curr > target_X:
        start_x_i = (X_i_curr - target_X) // 2
        img_cropped = img[:, start_x_i: start_x_i + target_X]
    else:
        img_cropped = img

    # Crop Image in Y (height) to match Vol
    target_Y = min(Y_v, img_cropped.shape[0])
    if img_cropped.shape[0] > target_Y:
        start_y_i = (img_cropped.shape[0] - target_Y) // 2
        img_cropped = img_cropped[start_y_i: start_y_i + target_Y, :]

    # Crop Volume in X if needed
    if X_v > target_X:
        start_x_v = (X_v - target_X) // 2
        vol_cropped = vol_scaled[start_x_v: start_x_v + target_X, :, :]
    else:
        vol_cropped = vol_scaled

    # Crop Volume in Y if needed
    if vol_cropped.shape[1] > target_Y:
        start_y_v = (vol_cropped.shape[1] - target_Y) // 2
        vol_cropped = vol_cropped[:, start_y_v: start_y_v + target_Y, :]

    # Check alignment
    if img_cropped.shape[1] != vol_cropped.shape[0] or img_cropped.shape[0] != vol_cropped.shape[1]:
        print(f"Shape mismatch! Img: {img_cropped.shape}, Vol: {vol_cropped.shape}")
        min_x = min(img_cropped.shape[1], vol_cropped.shape[0])
        min_y = min(img_cropped.shape[0], vol_cropped.shape[1])
        img_cropped = img_cropped[:min_y, :min_x]
        vol_cropped = vol_cropped[:min_x, :min_y, :]

    # Move img to CPU
    img_cropped_cpu = img_cropped.cpu()
    vol_cropped_cpu = vol_cropped  # Already on CPU

    # Normalize dtypes for saving/consumers
    return vol_cropped_cpu.float().contiguous(), img_cropped_cpu.float().contiguous()

def check_dimensions():
    """
    Checks dimensions of the first volume and image.
    """
    print("Checking dimensions...")
    
    # Path setup
    data_root = Path("data/raw")
    vol_path = data_root / "lightsheet_vol_6.9" / "Interp_Vol_ID_1.pt"
    img_path = data_root / "average_imgs" / "1scan (1).tif"
    
    if not vol_path.exists():
        print(f"Volume not found: {vol_path}")
        return
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
        
    # Read
    vol = read_volume(vol_path)
    img = read_image(img_path)
    
    print(f"Volume shape (X, Y, Z): {vol.shape}")
    print(f"Image shape (Y, X): {img.shape}")
    
    # Check against README
    # README says Vol: (74, 600, 100)
    # README says Img: (2448, 2048) -> (X=2448, Y=2048)
    
    # Report
    print("-" * 20)
    print("Comparison with README:")
    print(f"Volume X: Actual {vol.shape[0]} vs README 149")
    print(f"Volume Y: Actual {vol.shape[1]} vs README 600")
    print(f"Volume Z: Actual {vol.shape[2]} vs README 100")
    
    print(f"Image X: Actual {img.shape[1]} vs README 2448")
    print(f"Image Y: Actual {img.shape[0]} vs README 2048")
    print("-" * 20)


def preprocess_dataset(
    output_dir,
    input_dir,
    img_dir,
    downsampling_rate,
    scale_factor=8.0,
    crop_box_b: tuple[int, int, int, int] | None = None,
    crop_box_A: tuple[int, int, int, int, int, int] | None = None,
):
    """
    Preprocesses the dataset:
    1. Reads pairs from input_dir.
    2. Scales volume by scale_factor * downsampling_rate.
    3. Downsamples image by downsampling_rate.
    4. Crops to valid region.
    5. Saves as Ax=b pair in separate HDF5 files.
    """
    print("Starting preprocessing...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = find_raw_pairs(input_dir=input_dir, img_dir=img_dir)
    if not pairs:
        print(f"No matching raw pairs found in {input_dir} and {img_dir}")
        return

    print(f"Found {len(pairs)} pairs. Indices: {pairs[0].idx} to {pairs[-1].idx}")

    for pair in pairs:
        idx = pair.idx
        vol_path = pair.vol_path
        img_path = pair.img_path
            
        print(f"Processing index {idx}...")
        t0 = time.time()

        # Read + process to CPU tensors
        vol_cropped_cpu, img_cropped_cpu = preprocess_one_pair(
            vol_path=vol_path,
            img_path=img_path,
            downsampling_rate=downsampling_rate,
            scale_factor=scale_factor,
            crop_box_b=crop_box_b,
            crop_box_A=crop_box_A,
            device=device,
        )

        # Save to separate HDF5 file
        h5_path = output_dir / f"pair_{idx}.h5"

        save_pair_h5(h5_path=h5_path, A_cpu=vol_cropped_cpu, b_cpu=img_cropped_cpu)
            
        print(f"Saved {h5_path}. Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess paired volume/image data into per-index HDF5 files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/lightsheet_vol_6.9",
        help="Directory containing Interp_Vol_ID_*.pt volumes.",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="data/raw/20um_imgs",
        help="Directory containing 1scan (idx).tif images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/20um_imgs",
        help="Output directory for pair_{idx}.h5 files.",
    )
    parser.add_argument(
        "--downsampling-rate",
        type=float,
        default=0.125,
        help="Downsampling rate for the image; volume scaling uses scale_factor * downsampling_rate.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=8.0,
        help="Base scale factor applied to the volume before matching image space.",
    )
    parser.add_argument(
        "--skip-dimension-check",
        action="store_true",
        help="Skip the initial dimension sanity check.",
    )

    parser.add_argument(
        "--crop-box-b",
        type=str,
        default=None,
        help=(
            "Manual crop box for b in image coordinates, format 'x0,y0,x1,y1'. "
            "NOTE: x indexes the 2nd dim of b (width), y indexes the 1st dim (height)."
        ),
    )
    parser.add_argument(
        "--crop-box-a",
        type=str,
        default=None,
        help=(
            "Manual crop box for A in volume coordinates, format 'x0,y0,z0,x1,y1,z1' where A is (X,Y,Z). "
            "Must be consistent with crop_box_b via effective_vol_scale=scale_factor*downsampling_rate."
        ),
    )

    args = parser.parse_args()

    # Save into a subfolder of output_dir tagged by downsampling rate.
    # Example: data/processed/ds0p125
    ds_tag = f"ds{args.downsampling_rate:g}".replace(".", "p")
    args.output_dir = str(Path(args.output_dir) / ds_tag)

    if not args.skip_dimension_check:
        check_dimensions()

    crop_box_b = _parse_int_tuple(args.crop_box_b, n=4, name="crop_box_b") if args.crop_box_b else None
    crop_box_A = _parse_int_tuple(args.crop_box_a, n=6, name="crop_box_A") if args.crop_box_a else None

    preprocess_dataset(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        img_dir=args.img_dir,
        downsampling_rate=args.downsampling_rate,
        scale_factor=args.scale_factor,
        crop_box_b=crop_box_b,
        crop_box_A=crop_box_A,
    )