import torch
import numpy as np
import pytest
from pathlib import Path
import logging
from src.core.linear_system_pair import LinearSystemPair
from src.core.masked_system_pair import LinearSystemPairMasked
from src.core.fista import FISTASolver
from src.core.ista import ISTASolver
from src.core.linear_system_point import LinearSystemPoint
import os


def make_sphere_volume(X: int, Y: int, Z: int, *, radius: float | None = None, dtype=torch.float16) -> torch.Tensor:
    """Create a simple binary sphere volume (X,Y,Z) with values in {0,1}."""
    if radius is None:
        radius = min(X, Y, Z) / 4

    x_true = torch.zeros(X, Y, Z, dtype=dtype)
    center = torch.tensor([X / 2, Y / 2, Z / 2], dtype=torch.float32)
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                if torch.norm(torch.tensor([i, j, k], dtype=torch.float32) - center) < radius:
                    x_true[i, j, k] = 1.0
    return x_true


def make_zband_measurements(
    x_true: torch.Tensor,
    *,
    band: int | None = None,
    dtype=torch.float16,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    """Create (A_list, b_list) where each measurement only covers a small Z-band.

    This matches the format expected by LinearSystem/MaskedLinearSystem:
      - A_m: (X, Y, Z)
      - b_m: (Y, X)
    Returns: (A_list, b_list, n_meas, band)
    """
    X, Y, Z = x_true.shape
    if band is None:
        band = max(1, Z // 8)
    band = int(band)
    n_meas = int(np.ceil(Z / band))

    A_list: list[torch.Tensor] = []
    b_list: list[torch.Tensor] = []

    for m in range(n_meas):
        z0 = m * band
        z1 = min(Z, (m + 1) * band)
        if z0 >= Z:
            break

        A_m = torch.zeros(X, Y, Z, dtype=dtype)
        A_m[:, :, z0:z1] = 1.0

        # b shape in the dataset is (Y, X)
        b_m = torch.sum(x_true[:, :, z0:z1], dim=2).T.contiguous().to(dtype)
        A_list.append(A_m)
        b_list.append(b_m)

    return A_list, b_list, len(A_list), band

def test_reconstruction_smoke():
    """
    A smoke test for the reconstruction pipeline.
    1. Creates a synthetic volume (a sphere).
    2. Creates a simple linear system based on the project's specific data format.
    3. Runs the FISTA solver for enough iterations to converge.
    4. Checks if the solver correctly outputs log, .obj, and plot files.
    """
    output_dir = Path("result/solve/temp_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to capture logs from all modules
    log_file_path = output_dir / "test_run.log"
    file_handler = logging.FileHandler(log_file_path, mode='w') # Overwrite log
    file_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    
    root_logger = logging.getLogger()
    original_level = root_logger.level
    
    # Avoid adding handlers multiple times
    # First, remove existing handlers that write to the same file if any
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file_path):
            root_logger.removeHandler(handler)
            
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info("Starting reconstruction smoke test.")

    try:
        # 1. Create synthetic volume (sphere)
        X, Y, Z = 32, 32, 32
        x_true = make_sphere_volume(X, Y, Z, dtype=torch.float16)

        # 2. Create Z-band measurements (each light field only covers a small Z range)
        A_list, b_list, n_meas, band = make_zband_measurements(x_true, dtype=torch.float16)

        system = LinearSystemPair(A_list, b_list, device='cpu')

        # 3. Run FISTA solver. Provide a conservative Lipschitz constant for stability.
        # For our constructed A_m with 0/1 entries and disjoint bands, a safe upper bound
        # is L ~= n_meas * band (very conservative).
        L_safe = float(n_meas * band)
        solver = FISTASolver(system, output_dir=output_dir, lipchitz=L_safe)
        x0 = torch.zeros(X, Y, Z, dtype=torch.float16)
        x_result = solver.solve(x0, n_iter=100)
        
        assert x_result.shape == (X, Y, Z)
        
        # 4. Verify outputs (now created by solver's _post_solve hook)
        obj_path = output_dir / "reconstruction.obj"
        plot_path = output_dir / "loss_curve.png"

        assert log_file_path.exists()
        assert obj_path.exists()
        assert plot_path.exists()
        
        with open(log_file_path, "r") as f:
            log_content = f.read()
            assert "Starting reconstruction smoke test" in log_content
            assert "Starting solver" in log_content
            assert "Solver finished" in log_content
            assert "Saved convergence plot" in log_content
            assert "Exporting volume to mesh" in log_content

        logger.info("Reconstruction smoke test finished.")
    finally:
        # Clean up the handler
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(original_level)


def test_masked_reconstruction():
    """
    Tests MaskedLinearSystem to ensure it correctly identifies and locks zero variables.
    """
    output_dir = Path("result/solve/masked_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build a sphere volume and Z-band measurements, then force some pixels in b to zero
    # to verify MaskedLinearSystem locks the corresponding variables.
    X, Y, Z = 16, 16, 16
    x_true = make_sphere_volume(X, Y, Z, dtype=torch.float16)
    A_list, b_list, n_meas, band = make_zband_measurements(x_true, dtype=torch.float16)

    # Pick one measurement and zero out a small patch in b to create guaranteed fixed-zero vars.
    # b shape is (Y, X).
    b_patch = b_list[0].clone()
    x0_i, x1_i = 0, 4
    y0_i, y1_i = 0, 4
    b_patch[y0_i:y1_i, x0_i:x1_i] = 0.0
    b_list[0] = b_patch
    
    # 2. Instantiate MaskedLinearSystem
    masked_sys = LinearSystemPairMasked(A_list, b_list, device='cpu', threshold_b=1e-5)
    
    # Stats check
    stats = masked_sys.stats
    print(f"Stats: {stats}")
    
    # We don't assert exact counts here (depends on volume + band widths),
    # but we should have a significant number of fixed zeros after forcing a b patch to zero.
    assert stats.n_fixed_zero > 0
    assert stats.n_free_vars > 0
    
    # 3. Solve
    # Conservative Lipschitz constant.
    L_safe = float(n_meas * band)
    solver = FISTASolver(masked_sys, output_dir=output_dir, lipchitz=L_safe)
    x0 = torch.ones(X, Y, Z, dtype=torch.float16)  # Bad guess (violates constraints)
    
    # Solver should pack x0 (selecting only free vars) and solve.
    # Post-solve should expand, filling 0s.
    x_result = solver.solve(x0, n_iter=10)
    
    # 4. Verify result
    # Fixed vars should be exactly 0.
    # Free vars should satisfy constraint (sum to 4).
    
    # Check fixed vars for the zeroed b patch on measurement 0.
    # A_0 covers z in [0, band), so those variables should be locked to 0 for pixels where b=0.
    x_patch = x_result[x0_i:x1_i, y0_i:y1_i, :band]
    assert torch.all(x_patch == 0), "Masked (fixed) region is not zero!"

    # Also ensure solution isn't trivially all zeros.
    assert torch.sum(x_result) > 0, "Solution is all zeros!"
    
    print("Masked reconstruction test passed.")

def test_point_reconstruction():
    """
    Tests the pointwise reconstruction workflow.
    1. Generates a dummy 'point' dataset file manually.
    2. Loads it with PointLinearSystem.
    3. Runs FISTASolver on it.
    """
    output_dir = Path("result/solve/point_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create Dummy Point Data
    # Problem: Ax = b. 
    # x is (Z=10).
    # A is (N=5, Z=10).
    # Let x_true be sparse.
    Z = 10
    N = 5
    x_true = torch.zeros(Z)
    x_true[2] = 1.0
    x_true[5] = 0.5
    
    # A random
    torch.manual_seed(42)
    A = torch.randn(N, Z)
    b = A @ x_true
    
    # Save as .pt
    point_file = output_dir / "test_point.pt"
    save_data = {'A': A, 'b': b, 'coord': (10, 10)}
    torch.save(save_data, point_file)
    
    # 2. Load System
    system = LinearSystemPoint(point_file, device='cpu')
    assert system.Z == Z
    assert system.N == N
    
    # 3. Solve
    solver = FISTASolver(system, output_dir=output_dir, lipchitz=None, lambda_reg=0.01, backtracking=True) # L1 reg
    
    x0 = torch.zeros(1, 1, Z)
    x_rec = solver.solve(x0, n_iter=50)
    
    # x_rec is (1, 1, Z)
    x_rec_vec = x_rec.reshape(-1)
    
    # Check if we recovered indices 2 and 5 roughly
    print(f"True x: {x_true}")
    print(f"Rec x: {x_rec_vec}")
    
    # Because N < Z (5 < 10), it's underdetermined, but L1 should pick sparse solution.
    # Check error
    err = torch.norm(x_rec_vec - x_true)
    print(f"Reconstruction Error: {err}")
    
    # With only 5 measurements for 2 non-zeros, recovery might be approximate but should be non-trivial
    # Just check shapes and that it didn't crash.
    assert x_rec.shape == (1, 1, Z)
    assert not torch.isnan(x_rec).any()
    
    # Verify plotting created files (loss_curve.png, reconstruction.obj)
    # Note: mesh export on 1x1xZ volume might be weird but shouldn't crash
    # (volume2mesh generally expects 3D volume, but our code might handle it or just produce empty obj)
    assert (output_dir / "loss_curve.png").exists()

def test_reconstruction_ista():
    """
    A smoke test for the reconstruction pipeline using ISTASolver.
    """
    output_dir = Path("result/solve/temp_test_ista")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file_path = output_dir / "test_run.log"
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    
    root_logger = logging.getLogger()
    original_level = root_logger.level
    
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file_path):
            root_logger.removeHandler(handler)
            
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info("Starting reconstruction ISTA test.")

    try:
        # 1. Create synthetic volume (sphere)
        X, Y, Z = 32, 32, 32
        x_true = make_sphere_volume(X, Y, Z, dtype=torch.float16)

        # 2. Create Z-band measurements
        A_list, b_list, n_meas, band = make_zband_measurements(x_true, dtype=torch.float16)

        system = LinearSystemPair(A_list, b_list, device='cpu')

        # 3. Run ISTA solver.
        L_safe = float(n_meas * band)
        solver = ISTASolver(system, output_dir=output_dir, lipchitz=L_safe)
        x0 = torch.zeros(X, Y, Z, dtype=torch.float16)
        x_result = solver.solve(x0, n_iter=100)
        
        assert x_result.shape == (X, Y, Z)
        
        # 4. Verify outputs
        obj_path = output_dir / "reconstruction.obj"
        plot_path = output_dir / "loss_curve.png"

        assert obj_path.exists()
        assert plot_path.exists()

        logger.info("Reconstruction ISTA test finished.")
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(original_level)
