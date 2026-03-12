import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to the .pt file")
    parser.add_argument("--save", help="Save the histogram as a file instead of showing it", default=None)
    args = parser.parse_args()

    # Try to use an interactive backend if we're not saving
    if not args.save:
        try:
            # Common interactive backends
            for backend in ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg', 'GTK3Agg']:
                try:
                    matplotlib.use(backend)
                    break
                except:
                    continue
        except:
            pass

    try:
        data = torch.load(args.file_path, map_location='cpu')
        
        if isinstance(data, dict):
            vol = data.get('reconstruction')
            if vol is None:
                vol = data.get('A')
        elif isinstance(data, torch.Tensor):
            vol = data
        else:
            print(f"Error: Unknown data type {type(data)}")
            sys.exit(1)
            
        if vol is None:
            print("Error: Could not find volume data in the file.")
            sys.exit(1)
            
        vol_np = vol.numpy()
        
        # 展平成1D
        pixels = vol_np.flatten()

        # 统计直方图
        hist, _ = np.histogram(pixels, bins=256, range=(0,256))

        plt.figure(figsize=(8,5))
        plt.bar(range(256), hist)

        plt.xlabel("Gray Value")
        plt.ylabel("Pixel Count")
        plt.title(f"Histogram of {os.path.basename(args.file_path)}")

        plt.ylim(0,1000)

        plt.tight_layout()
        
        if args.save:
            plt.savefig(args.save)
            print(f"Histogram saved to {args.save}")
        else:
            print("Displaying histogram... (requires X11)")
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
