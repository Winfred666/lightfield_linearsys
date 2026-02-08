import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Config
base_dir = Path("result/solve_point/newton_crop_20um_0p25")
regs = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

means = []
valid_regs = []

print("Analyzing regression sweep results...")

for reg in regs:
    reg_str = f"{reg:.1e}"
    reg_dir = base_dir / f"reg_{reg_str}"
    
    if not reg_dir.exists():
        print(f"Directory not found: {reg_dir}")
        continue
        
    # Find latest timestamp folder
    # We might need to wait if the folder is just being created? 
    # But for analysis we assume it's there or we skip.
    subdirs = sorted([d for d in reg_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not subdirs:
        print(f"No timestamp subdir in {reg_dir}")
        continue
    
    # Pick the latest one
    latest_dir = subdirs[-1]
    loss_dir = latest_dir / "loss_curve"
    
    if not loss_dir.exists():
        print(f"No loss_curve dir in {latest_dir}")
        continue
        
    txt_files = list(loss_dir.glob("residual_norm_*.txt"))
    if not txt_files:
        print(f"No residual txt files in {loss_dir}")
        continue
        
    final_residuals = []
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    final_val = float(lines[-1].strip())
                    final_residuals.append(final_val)
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            
    if final_residuals:
        mean_res = np.mean(final_residuals)
        means.append(mean_res)
        valid_regs.append(reg)
        print(f"Reg {reg_str}: Mean Residual = {mean_res:.6e} (from {len(final_residuals)} batches)")
    else:
        print(f"Reg {reg_str}: No valid data extracted")

if not valid_regs:
    print("No valid data found to plot.")
    exit(1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(valid_regs, means, marker='o', linestyle='-')
plt.xscale('log')
plt.yscale('log') # Residuals can vary by orders of magnitude
plt.xlabel('Regularization Lambda (log scale)')
plt.ylabel('Mean Final Residual Norm (log scale)')
plt.title('Residual Norm vs Lambda_reg')
plt.grid(True, which="both", ls="-", alpha=0.5)

out_path = base_dir / "reg_sweep_plot.png"
plt.savefig(out_path)
print(f"Saved plot to {out_path}")
