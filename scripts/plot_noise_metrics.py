import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Configuration
sigmas = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 10.0]
# Treat sigmas as strings for paths, floats for plotting
sigma_strs = ["0.01", "0.1", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0", "10.0"]

cases = [
    {
        "name": "Case 1",
        "id": "case_1",
        "clean_tag": "case_1",
        "recon_base": "result/solve_point/newton_balls_1"
    },
    {
        "name": "Case 2",
        "id": "case_2",
        "clean_tag": "case_2",
        "recon_base": "result/solve_point/newton_balls_2"
    }
]

def parse_psnr(metrics_file):
    if not metrics_file.exists():
        return None
    with open(metrics_file, "r") as f:
        content = f.read()
        # Look for "PSNR: 30.50 dB"
        match = re.search(r"PSNR:\s+([0-9.]+)\s+dB", content)
        if match:
            return float(match.group(1))
    return None

def main():
    output_dir = Path("result/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        print(f"Plotting {case['name']}...")
        
        psnr_clean = []
        psnr_noisy = []
        valid_sigmas = []

        recon_base = Path(case["recon_base"])
        
        for s_str, s_val in zip(sigma_strs, sigmas):
            recon_dir = recon_base / f"noise_sigma_{s_str}"
            
            # Helper to find latest metrics file matching pattern
            def find_metrics(parent_dir, tag):
                # Look for **/compare_{tag}/metrics.txt
                # parent_dir is .../noise_sigma_0.01
                # structure: .../noise_sigma_0.01/<timestamp>/compare_{tag}/metrics.txt
                candidates = list(parent_dir.glob(f"**/compare_{tag}/metrics.txt"))
                if not candidates:
                    return None
                # Return newest
                return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

            # 1. Clean Metrics
            clean_file = find_metrics(recon_dir, case['clean_tag'])
            p_clean = parse_psnr(clean_file) if clean_file else None
            
            # 2. Noisy Metrics
            noisy_tag = f"sigma_{s_str}"
            noisy_file = find_metrics(recon_dir, noisy_tag)
            p_noisy = parse_psnr(noisy_file) if noisy_file else None
            
            if p_clean is not None and p_noisy is not None:
                psnr_clean.append(p_clean)
                psnr_noisy.append(p_noisy)
                valid_sigmas.append(s_val)
            else:
                print(f"Missing metrics for sigma={s_str} in {case['name']}")
                
        if not valid_sigmas:
            print(f"No valid data for {case['name']}. Skipping plot.")
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(valid_sigmas, psnr_clean, marker='o', label='PSNR vs Clean GT')
        plt.plot(valid_sigmas, psnr_noisy, marker='x', linestyle='--', label='PSNR vs Noisy GT')
        
        plt.title(f"PSNR vs Noise Level ({case['name']})")
        plt.xlabel("Sigma (Noise Level)")
        plt.ylabel("PSNR (dB)")
        plt.grid(True)
        plt.legend()
        
        out_path = output_dir / f"noise_metrics_{case['id']}.png"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        plt.close()

if __name__ == "__main__":
    main()