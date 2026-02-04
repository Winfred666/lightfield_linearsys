import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def parse_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if "PSNR:" in line:
                metrics['psnr'] = float(line.split(":")[1].replace("dB", "").strip())
            if "MSE:" in line:
                metrics['mse'] = float(line.split(":")[1].strip())
    return metrics

def plot_reg_sweep():
    base_dir = Path("result/solve_point/newton_balls_2/reg")
    regs = [10**(-i) for i in range(1, 10)]
    
    psnr_clean = []
    psnr_noisy = []
    valid_regs = []
    
    for reg in regs:
        reg_str = f"{reg:.1e}"
        reg_dir = base_dir / f"reg_{reg_str}"
        
        if not reg_dir.exists():
            continue
            
        # Find latest timestamp
        subdirs = sorted([d for d in reg_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        if not subdirs:
            continue
        latest_dir = subdirs[0]
        
        # Check for metrics
        # Clean GT tag: compare_case_2 (from folder name data/synthetic/balls/case_2)
        # Noisy GT tag: compare_sigma_2.0 (from folder name data/synthetic/balls/case_2_noise/sigma_2.0)
        
        clean_metrics_path = latest_dir / "compare_case_2" / "metrics.txt"
        noisy_metrics_path = latest_dir / "compare_sigma_2.0" / "metrics.txt"
        
        if clean_metrics_path.exists() and noisy_metrics_path.exists():
            m_clean = parse_metrics(clean_metrics_path)
            m_noisy = parse_metrics(noisy_metrics_path)
            
            psnr_clean.append(m_clean['psnr'])
            psnr_noisy.append(m_noisy['psnr'])
            valid_regs.append(reg)
            
    if not valid_regs:
        print("No valid metrics found.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    # Match style with scripts/plot_noise_metrics.py
    plt.semilogx(valid_regs, psnr_clean, marker='o', label='PSNR vs Clean GT')
    plt.semilogx(valid_regs, psnr_noisy, marker='x', linestyle='--', label='PSNR vs Noisy GT')
    
    plt.xlabel('Regularization Lambda (log scale)')
    plt.ylabel('PSNR (dB)')
    plt.title('Reconstruction Quality vs Regularization')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    out_path = base_dir / "reg_sweep_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_reg_sweep()
