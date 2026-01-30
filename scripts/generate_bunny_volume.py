import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def generate_gt_volume(bunny_path, ref_vol_path, output_path):
    print(f"Loading bunny SDF from {bunny_path}...")
    sdf = np.load(bunny_path) # (128, 128, 128)
    
    # Convert SDF to binary density (1.0 inside, 0.0 outside)
    # "inside mesh is negative"
    mask = (sdf < 0).astype(np.float32)
    mask_t = torch.from_numpy(mask) # (128, 128, 128)
    
    print(f"Loading reference volume for shape from {ref_vol_path}...")
    ref_vol = torch.load(ref_vol_path, map_location='cpu')
    target_shape = ref_vol.shape # (X, Y, Z) = (149, 600, 100)
    print(f"Target shape: {target_shape}")
    
    target_X, target_Y, target_Z = target_shape
    
    # We want to embed the bunny into this volume.
    # Bunny is 128x128x128.
    # Target min dim is Z=100.
    # Let's scale bunny to fit within Z=90 (leave some padding).
    
    bunny_size = 90
    
    # Interpolate bunny to (bunny_size, bunny_size, bunny_size)
    # F.interpolate expects (N, C, D, H, W) -> (N, C, Z, Y, X) 
    # Our mask is (X, Y, Z)? Or (Z, Y, X)? 
    # Usually 3D arrays are (D, H, W) -> (Z, Y, X) in pytorch convention for interpolation
    # But let's assume the bunny is isotropic.
    
    mask_in = mask_t.unsqueeze(0).unsqueeze(0) # (1, 1, 128, 128, 128)
    
    mask_scaled = F.interpolate(
        mask_in, 
        size=(bunny_size, bunny_size, bunny_size),
        mode='trilinear',
        align_corners=False
    )
    
    # Threshold back to binary after interpolation
    mask_scaled = (mask_scaled > 0.5).float()
    
    mask_scaled = mask_scaled.squeeze(0).squeeze(0) # (90, 90, 90)
    
    # Create empty target volume
    gt_vol = torch.zeros(target_shape, dtype=torch.float32)
    
    # Calculate center offsets
    start_x = (target_X - bunny_size) // 2
    start_y = (target_Y - bunny_size) // 2
    start_z = (target_Z - bunny_size) // 2
    
    print(f"Placing bunny at offset ({start_x}, {start_y}, {start_z})")
    
    gt_vol[
        start_x : start_x + bunny_size,
        start_y : start_y + bunny_size,
        start_z : start_z + bunny_size
    ] = mask_scaled
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving GT volume to {output_path}...")
    torch.save(gt_vol, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bunny-path", default="data/raw/bunny/bunny_128x128x128.npy")
    parser.add_argument("--ref-vol-path", default="data/raw/lightsheet_vol_6.9/Interp_Vol_ID_1.pt")
    parser.add_argument("--output-path", default="data/synthetic/bunny/gt_volume.pt")
    args = parser.parse_args()
    
    generate_gt_volume(args.bunny_path, args.ref_vol_path, args.output_path)
