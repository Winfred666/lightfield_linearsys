import torch
import pytest
from src.core.fista import FISTASolver
from src.core.linear_system_pair import LinearSystemPair
from src.core.masked_system_pair import LinearSystemPairMasked
from pathlib import Path

def test_linear_system_setup_and_filter():
    # Synthetic data: X=2, Y=2, Z=2
    # 4 pixels total.
    # Create 2 measurements.
    
    # Measurement 1:
    # Pixel (0,0): valid
    # Pixel (0,1): valid
    # Pixel (1,0): zero (should be filtered)
    # Pixel (1,1): valid
    
    A1 = torch.zeros(2, 2, 2)
    A1[0, 0, :] = torch.tensor([1.0, 1.0])
    A1[0, 1, :] = torch.tensor([0.5, 0.5])
    # A1[1, 0] is 0
    A1[1, 1, :] = torch.tensor([2.0, 2.0])
    
    b1 = torch.ones(2, 2) # (Y, X) usually, let's say b matches (Y, X). Here X=Y=2.
    
    # Measurement 2:
    # All valid
    A2 = torch.ones(2, 2, 2)
    b2 = torch.ones(2, 2)
    
    system = LinearSystemPair([A1, A2], [b1, b2], device='cpu')
    
    # LinearSystem filters by rows whose max(|A_row|) > threshold_A.
    # In A1, pixel (1,0) is all zeros -> filtered. So:
    # Measurement 1 keeps 3 rows, measurement 2 keeps 4 rows => 7 total.
    assert system.valid_indices.shape[0] == 7
    assert system.valid_A.shape[0] == 7
    assert system.b.shape[0] == 7

def test_fista_convergence_tiny(tmp_path: Path):
    # 1 pixel problem: X=1, Y=1, Z=2
    # x_true = [1.0, 0.0]
    # A = [[1.0, 0.0], [0.0, 1.0]] (Two measurements)
    # b = [1.0, 0.0]
    
    # Measurement 1: A=[1, 0], b=1
    A1 = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
    b1 = torch.tensor([[1.0]])        # (1, 1)
    
    # Measurement 2: A=[0, 1], b=0
    A2 = torch.tensor([[[0.0, 1.0]]])
    b2 = torch.tensor([[0.0]])
    
    system = LinearSystemPair([A1, A2], [b1, b2])
    
    # Solve
    solver = FISTASolver(system, output_dir=tmp_path, lambda_reg=0.01) # Small reg
    x0 = torch.zeros(1, 1, 2)
    
    x_est = solver.solve(x0, n_iter=100, verbose=False)
    
    # Expected: x approx [1, 0]
    print(f"Estimated x: {x_est}")
    
    assert torch.allclose(x_est, torch.tensor([[[1.0, 0.0]]]), atol=0.1)


def test_masked_system_infers_zero_var(tmp_path: Path):
    # 1 pixel, 2 unknowns (Z=2)
    # Measurement 1 constrains x0 via b=1
    # Measurement 2 has b=0 but A=[0,1] => forces x1=0 under x>=0

    A1 = torch.tensor([[[1.0, 0.0]]])
    b1 = torch.tensor([[1.0]])
    A2 = torch.tensor([[[0.0, 1.0]]])
    b2 = torch.tensor([[0.0]])
    system = LinearSystemPairMasked([A1, A2], [b1, b2], device='cpu')

    solver = FISTASolver(system, output_dir=tmp_path, lambda_reg=0.01)
    x0 = torch.zeros(1, 1, 2)
    x_est = solver.solve(x0, n_iter=80, verbose=False)

    # Still solves full x but should have inferred x1 fixed 0.
    assert x_est.shape == (1, 1, 2)
    assert torch.allclose(x_est[..., 1], torch.zeros_like(x_est[..., 1]), atol=1e-4)


def test_fista_convergence_random(tmp_path: Path):
    # Larger random problem
    torch.manual_seed(42)
    X, Y, Z = 5, 5, 10
    
    x_true = torch.rand(X, Y, Z) * (torch.rand(X, Y, Z) > 0.8).float() # Sparse x
    
    # Create 2 random measurements
    A1 = torch.randn(X, Y, Z)
    A2 = torch.randn(X, Y, Z)
    
    # Generate b = Ax (forward pass simulation)
    # We can use the system logic manually
    # y = sum(A * x, dim=-1)
    b1 = torch.sum(A1 * x_true, dim=-1).T # Transpose to (Y, X)
    b2 = torch.sum(A2 * x_true, dim=-1).T
    
    system = LinearSystemPair([A1, A2], [b1, b2])
    
    solver = FISTASolver(system, output_dir=tmp_path, lambda_reg=0.0, lipchitz=None) # No reg for exact match test
    
    # Estimate L: max eigenvalue of A^T A roughly
    # Or just run enough iters with small step
    solver.L = 100.0 # Conservative estimate
    
    x0 = torch.zeros(X, Y, Z)
    x_est = solver.solve(x0, n_iter=200, verbose=False)
    
    # Check residual
    # Since we have 2 measurements for Z=10 unknowns, it's underdetermined (2 equations per pixel).
    # We can't recover x_true exactly without sparsity constraints and enough measurements.
    # But we should minimize ||Ax - b||.
    
    pred = system.forward(x_est)
    loss = torch.norm(pred - system.b)
    
    print(f"Final residual norm: {loss.item()}")
    
    # Initial residual
    pred0 = system.forward(x0)
    loss0 = torch.norm(pred0 - system.b)
    print(f"Initial residual norm: {loss0.item()}")
    
    assert loss < loss0 * 0.1 # Should reduce error significantly



