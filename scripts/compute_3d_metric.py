import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import numpy as np

def compute_metrics(recon_path, gt_path):
    print(f"Loading reconstruction from {recon_path}...")
    recon_data = torch.load(recon_path, map_location='cpu')
    if isinstance(recon_data, dict) and 'reconstruction' in recon_data:
        recon = recon_data['reconstruction']
    else:
        recon = recon_data
    
    print(f"Loading GT from {gt_path}...")
    gt = torch.load(gt_path, map_location='cpu')
    
    # Ensure shapes match
    if recon.shape != gt.shape:
        print(f"Shape mismatch: Recon {recon.shape} vs GT {gt.shape}. Truncating/Padding not implemented yet.")
        return

    # Convert to float
    recon = recon.float()
    gt = gt.float()
    
    # MSE
    mse = F.mse_loss(recon, gt).item()
    
    # PSNR
    # Dynamic range: GT is 0 or 1.
    data_range = 1.0 
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
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recon_path", help="Path to reconstruction.pt")
    parser.add_argument("--gt-path", default="data/synthetic/bunny/gt_volume.pt")
    args = parser.parse_args()
    
    compute_metrics(args.recon_path, args.gt_path)
