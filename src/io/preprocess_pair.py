import torch
from pathlib import Path
import sys
import os
import h5py
import numpy as np
import argparse

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.io.readers import read_volume, read_image, scale_volume
import time

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


def preprocess_dataset(output_dir="data/processed", scale_factor=8.0):
    """
    Preprocesses the dataset:
    1. Reads pairs.
    2. Scales volume.
    3. Crops to valid region.
    4. Saves as Ax=b pair in separate HDF5 files.
    """
    print(f"Starting preprocessing with scale_factor={scale_factor}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    data_root = Path("data/raw")
    vol_dir = data_root / "lightsheet_vol_6.9"
    img_dir = data_root / "average_imgs"
    
    # Range 1 to 121
    for idx in range(1, 122):
        vol_name = f"Interp_Vol_ID_{idx}.pt"
        img_name = f"1scan ({idx}).tif"
        
        vol_path = vol_dir / vol_name
        img_path = img_dir / img_name
        
        if not vol_path.exists() or not img_path.exists():
            print(f"Missing pair for index {idx}. Skipping.")
            continue
            
        print(f"Processing index {idx}...")
        t0 = time.time()
        
        # Read
        vol = read_volume(vol_path) # (X, Y, Z)
        img = read_image(img_path) # (Y, X) numpy
        img = torch.from_numpy(img).float()
        
        # Move to device
        vol = vol.to(device)
        img = img.to(device)
        
        # Optimization: Crop Volume BEFORE scaling to save memory.
        # We know we need final Y = 2048. Scale factor = 8.
        # So we need input Y slice of 2048 / 8 = 256.
        # Original Y = 600.
        
        # Dimensions check
        X_v_orig, Y_v_orig, Z_v_orig = vol.shape
        Y_i_orig, X_i_orig = img.shape
        
        # Calculate target crop size in Low Res (Volume) domain for Y
        # We want to match Image Y (2048) or Vol Scaled Y (4800). Min is 2048.
        target_Y_highres = min(Y_i_orig, int(Y_v_orig * scale_factor))
        target_Y_lowres = int(target_Y_highres / scale_factor)
        
        if target_Y_lowres < Y_v_orig:
            start_y_v = (Y_v_orig - target_Y_lowres) // 2
            vol = vol[:, start_y_v : start_y_v + target_Y_lowres, :]
            
        # Chunked Scaling on GPU to avoid OOM
        Z_chunks = []
        with torch.no_grad():
            for i in range(4):
                z_start_in = i * 25
                # For the first 3 chunks, we need +1 overlap for correct interpolation at the boundary
                if i < 3:
                    z_end_in = (i + 1) * 25 + 1
                else:
                    z_end_in = 100
                    
                # Slice input (already cropped in Y)
                vol_chunk = vol[:, :, z_start_in : z_end_in] 
                
                # Scale
                vol_chunk_scaled = scale_volume(vol_chunk, scale_factor=scale_factor)
                
                # Crop output to target Z size (200)
                target_z_chunk = 200
                vol_chunk_scaled = vol_chunk_scaled[:, :, :target_z_chunk]
                
                # Move to CPU immediately
                Z_chunks.append(vol_chunk_scaled.cpu())
                
                del vol_chunk
                del vol_chunk_scaled
                torch.cuda.empty_cache()
        
        # Concatenate on CPU
        vol_scaled = torch.cat(Z_chunks, dim=2) # On CPU
        
        # Now dimensions
        X_v, Y_v, Z_v = vol_scaled.shape
        
        # Target Crop Size for X
        # Image X: 2448. Vol Scaled X: 1192. Min is 1192.
        target_X = min(X_v, X_i_orig)
        
        # Crop Image in X (width) to match Vol
        if X_i_orig > target_X:
            start_x_i = (X_i_orig - target_X) // 2
            img_cropped = img[:, start_x_i : start_x_i + target_X]
        else:
            img_cropped = img
            
        # Crop Image in Y (height) to match Vol
        target_Y = min(Y_v, img_cropped.shape[0])
        if img_cropped.shape[0] > target_Y:
            start_y_i = (img_cropped.shape[0] - target_Y) // 2
            img_cropped = img_cropped[start_y_i : start_y_i + target_Y, :]
            
        # Crop Volume in X if needed
        if X_v > target_X:
             start_x_v = (X_v - target_X) // 2
             vol_cropped = vol_scaled[start_x_v : start_x_v + target_X, :, :]
        else:
             vol_cropped = vol_scaled

        # Crop Volume in Y if needed
        if vol_cropped.shape[1] > target_Y:
             start_y_v = (vol_cropped.shape[1] - target_Y) // 2
             vol_cropped = vol_cropped[:, start_y_v : start_y_v + target_Y, :]
             
        # Check alignment
        if img_cropped.shape[1] != vol_cropped.shape[0] or img_cropped.shape[0] != vol_cropped.shape[1]:
             print(f"Shape mismatch! Img: {img_cropped.shape}, Vol: {vol_cropped.shape}")
             min_x = min(img_cropped.shape[1], vol_cropped.shape[0])
             min_y = min(img_cropped.shape[0], vol_cropped.shape[1])
             img_cropped = img_cropped[:min_y, :min_x]
             vol_cropped = vol_cropped[:min_x, :min_y, :]

        # Move img to CPU
        img_cropped_cpu = img_cropped.cpu()
        vol_cropped_cpu = vol_cropped # Already on CPU
        
        # Save to separate HDF5 file
        # Using format pair_001.h5 for easier sorting/globbing, or just pair_1.h5 to match current scheme
        # Let's stick to pair_{idx}.h5 to minimize friction
        h5_path = output_dir / f"pair_{idx}.h5"
        
        with h5py.File(h5_path, 'w') as f:
            # Chunking strategies:
            # A: (X, Y, Z). We often read sub-regions in X,Y. Z is relatively small (200).
            # We want to enable fast spatial crop reading.
            # Chunk size: (32, 32, Z) seems reasonable for spatial locality.
            chunks_A = (32, 32, vol_cropped_cpu.shape[2])
            chunks_b = (32, 32)
            
            f.create_dataset('A', data=vol_cropped_cpu.numpy(), chunks=chunks_A, dtype='f4')
            f.create_dataset('b', data=img_cropped_cpu.numpy(), chunks=chunks_b, dtype='f4')
            
        print(f"Saved {h5_path}. Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset with scaling.")
    parser.add_argument("--scale_factor", type=float, default=8.0, help="Scaling factor for volume interpolation.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. If not specified, defaults to data/processed_scale_{scale_factor}.")
    args = parser.parse_args()
    
    # Logic for default output dir
    if args.output_dir is None:
        if args.scale_factor == 8.0:
             # Keep backward compatibility for default scale
             args.output_dir = "data/processed"
        else:
             args.output_dir = f"data/processed_scale_{args.scale_factor}"

    # check_dimensions() 
    preprocess_dataset(output_dir=args.output_dir, scale_factor=args.scale_factor)
