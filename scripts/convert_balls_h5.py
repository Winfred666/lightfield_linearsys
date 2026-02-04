import h5py
import torch
import os
import argparse
from pathlib import Path




def convert_h5_to_pt(h5_path, output_path, index=0, sigma=0.0):
    print(f"Converting {h5_path} to {output_path} (index={index})...")
    with h5py.File(h5_path, 'r') as f:
        # Expected shape (500, 64, 64, 64)
        vol_np = f['volumes'][index]
        vol = torch.from_numpy(vol_np).float()

    # Record original range (from the clean volume) so we can keep it consistent after noise
    orig_min = float(vol.min().item())
    orig_max = float(vol.max().item())

    if sigma > 0.0:
        print(f"Adding Gaussian noise with sigma={sigma}...")
        noise = torch.randn_like(vol) * sigma
        vol = vol + noise
        # Keep values within original density range to avoid changing min/max due to noise
        vol = torch.clamp(vol, min=orig_min, max=orig_max)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(vol, output_path)
    print(f"Saved to {output_path} (Shape: {vol.shape}, Range: [{vol.min().item():.6g}, {vol.max().item():.6g}])")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sigma", type=float, default=0.0, help="Standard deviation of Gaussian noise to add.")
    args = parser.parse_args()
    
    convert_h5_to_pt(args.h5_path, args.output_path, sigma=args.sigma)
