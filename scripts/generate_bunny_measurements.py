import math
import torch
from pathlib import Path
import argparse
import sys
import os
import imageio
import re

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def generate_2d_matern_grf(
    grid_size,
    alpha=5.0,
    tau=2.0,
    sigma=None,
    seed=None,
    device=None
):
    """
    Generate a 2D Matérn-type Gaussian Random Field (periodic) using spectral synthesis.

    Parameters
    ----------
    grid_size : int
        H = W of the square grid.
    alpha : float
        Smoothness exponent. Larger alpha ⇒ smoother field.
    tau : float
        Inverse correlation length; Large τ → small correlation length → rapid decorrelation → much rougher field
    sigma : float or None
        Overall amplitude. If None, use same default as common PDE libraries:
            sigma = tau**(0.5*(2*alpha - 1))
    seed : int or None
        Reproducible sampling.
    device : torch.device or None
        CPU/CUDA.
    Returns
    -------
    field : torch.Tensor, shape [H, W], dtype float32
        Real-valued Matérn GRF.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # default amplitude rule (same as GaussianRF you showed)
    if sigma is None:
        sigma = tau**(0.5*(2*alpha - 1))

    # ----------------------------
    # 1. Frequency grid (2D, radians)
    # ----------------------------
    kx = 2.0 * math.pi * torch.fft.fftfreq(grid_size, device=device)
    ky = 2.0 * math.pi * torch.fft.fftfreq(grid_size, device=device)

    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K2 = KX.square() + KY.square()

    # ----------------------------
    # 2. Power spectrum for Matérn
    #    S(k) = sigma^2 * (k^2 + tau^2)^(-alpha)
    # ----------------------------
    sqrt_spec = sigma * (K2 + tau**2).pow(-alpha / 2.0)
    sqrt_spec[0, 0] = 0.0  # remove mean mode

    # ----------------------------
    # 3. Real-valued white noise → FFT
    # ----------------------------
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        noise = torch.randn((grid_size, grid_size), generator=g, device=device)
    else:
        noise = torch.randn((grid_size, grid_size), device=device)

    noise_ft = torch.fft.fft2(noise)

    # ----------------------------
    # 4. Apply spectral filter
    # ----------------------------
    filtered_ft = noise_ft * sqrt_spec

    # ----------------------------
    # 5. Inverse FFT → Real Matérn field
    # ----------------------------
    field = torch.fft.ifft2(filtered_ft).real.to(torch.float32)

    # normalize to 0-1.0 like density field
    field_min = field.min()
    field_max = field.max()
    field = (field - field_min) / (field_max - field_min + 1e-8) * 1.0
    return field

def numerical_sort_key(p: Path):
    numbers = re.findall(r'\d+', p.name)
    if numbers:
        return int(numbers[-1])
    return p.name

def generate_measurements(gt_vol_path, raw_A_dir, output_dir, *, noise_index: int = 0):
    print(f"Loading GT volume from {gt_vol_path}...")
    gt_vol = torch.load(gt_vol_path, map_location='cpu') # (X, Y, Z)
    print(f"GT Volume shape: {gt_vol.shape}")
    
    raw_A_dir = Path(raw_A_dir)
    output_dir = Path(output_dir)
    if noise_index is not None:
        output_dir = output_dir / f"noise_m{int(noise_index):04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all A files
    files = sorted(list(raw_A_dir.glob("Interp_Vol_ID_*.pt")), key=numerical_sort_key)
    print(f"Found {len(files)} A matrices.")
    
    for i, f_path in enumerate(files):
        if i % 10 == 0:
            print(f"Processing {i}/{len(files)}: {f_path.name}")
        # Extract ID
        # Format: Interp_Vol_ID_{idx}.pt
        idx = int(f_path.stem.split('_')[-1])
        
        A = torch.load(f_path, map_location='cpu') # (X, Y, Z)
        
        if A.shape != gt_vol.shape:
            print(f"Warning: Shape mismatch for {f_path.name}: {A.shape} vs {gt_vol.shape}. Skipping.")
            continue
            
        # Compute projection: b = sum(A * x, dim=Z)
        # Result is (X, Y)
        b_xy = torch.sum(A * gt_vol, dim=2)
        
        # Convert to (Y, X) for image format
        b_yx = b_xy.T # (Y, X)

        # Apply Matérn GRF noise to one selected measurement (by index in the loop).
        if int(i) == int(noise_index):
            print(f"Applying Matérn GRF noise to measurement i={i} (file idx={idx})...")
            
            H, W = b_yx.shape
            grid_size = max(H, W)
            
            # Noise hyperparams are intentionally fixed / hard-coded.
            noise_field = generate_2d_matern_grf(
                grid_size=grid_size,
                alpha=20.0,
                tau=0.6,
                device=b_yx.device
            )
            
            # Crop if necessary
            noise_field = noise_field[:H, :W]
            
            # SNR ~ 0.1 interpretation:
            # Assuming this means "Noise Level" is 0.1 (i.e., 10% of signal max)
            # Since true SNR=0.1 would mean signal is buried in noise.
            noise_scale = 0.1 * b_yx.max()
            print(f"Adding noise with scale {noise_scale:.4f} (Max signal: {b_yx.max():.4f})")
            
            noise_toadd = noise_field * noise_scale
            b_yx_noisy = b_yx + noise_toadd
            
            # Save debug for visualization
            debug_path = output_dir / f"debug_measurement_{idx}.pt"
            torch.save((A, b_yx_noisy), debug_path)
            print(f"Saved debug measurement (A, b_noisy) to {debug_path}")
            
            b_yx = b_yx_noisy
        
        # Save as TIFF
        # Filename format: "1scan ({idx}).tif"
        out_name = f"1scan ({idx}).tif"
        out_path = output_dir / out_name
        
        # Convert to numpy and save
        b_np = b_yx.numpy().astype('float32')
        imageio.imwrite(out_path, b_np)

    print(f"Saved {len(files)} measurements to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-vol-path", default="data/synthetic/bunny/gt_volume.pt")
    parser.add_argument("--raw-A-dir", default="data/raw/lightsheet_vol_6.9")
    parser.add_argument("--output-dir", default="data/synthetic/bunny/measurements")
    parser.add_argument(
        "--noise-index",
        type=int,
        default=0,
        help=(
            "Which measurement (0-based index in sorted Interp_Vol_ID_*.pt files) to inject Matérn GRF noise into. "
            "The output dir will be suffixed with noise_mXXXX."
        ),
    )
    args = parser.parse_args()
    
    generate_measurements(args.gt_vol_path, args.raw_A_dir, args.output_dir, noise_index=args.noise_index)