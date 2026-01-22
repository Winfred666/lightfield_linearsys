import torch
import sys

try:
    # Try to find a processed file
    import glob
    files = glob.glob("data/processed/pair_*.pt")
    if files:
        f = files[0]
        print(f"Loading {f}")
        data = torch.load(f)
    else:
        print("No processed files found, falling back to raw")
        data = torch.load("data/raw/lightsheet_vol_6.9/Interp_Vol_ID_1.pt")

    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"Key: {k}, Type: {type(v)}")
            if hasattr(v, 'shape'):
                print(f"  Shape: {v.shape}")
                print(f"  Dtype: {v.dtype}")
                if isinstance(v, torch.Tensor):
                    if k == 'A':
                        # A shape (X, Y, Z). Row = (x, y)
                        norms = torch.norm(v.reshape(-1, v.shape[2]), dim=1)
                        print(f"  Row norms: min={norms.min()}, max={norms.max()}, mean={norms.mean()}")
                        print(f"  Quantiles: 10%={torch.quantile(norms, 0.1)}, 50%={torch.quantile(norms, 0.5)}, 90%={torch.quantile(norms, 0.9)}")
                        print(f"  < 0.1: {torch.sum(norms < 0.1).item()}/{norms.numel()}")
                        print(f"  < 1.0: {torch.sum(norms < 1.0).item()}/{norms.numel()}")
    elif isinstance(data, torch.Tensor):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    else:
        print(f"Content: {data}")
except Exception as e:
    print(f"Error: {e}")
