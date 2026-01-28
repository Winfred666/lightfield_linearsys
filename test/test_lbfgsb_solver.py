import torch
import pytest
from LF_linearsys.core.l_bfgs_b import LBFGSBSolver
from LF_linearsys.core.linear_system_pair import LinearSystemPair
from pathlib import Path
import logging

def test_lbfgsb_convergence_tiny(tmp_path: Path):
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
    solver = LBFGSBSolver(system, output_dir=tmp_path, lambda_reg=0.0) 
    x0 = torch.zeros(1, 1, 2)
    
    # Use few iterations, L-BFGS-B should converge very fast on this quadratic problem
    x_est = solver.solve(x0, n_iter=20, verbose=False)
    
    # Expected: x approx [1, 0]
    print(f"Estimated x: {x_est}")
    
    assert torch.allclose(x_est, torch.tensor([[[1.0, 0.0]]]), atol=1e-4)

def test_lbfgsb_non_negativity(tmp_path: Path):
    # Problem where least squares solution is negative, but constrained solution should be 0.
    # min (x + 1)^2  => x = -1 unconstrained.
    # s.t. x >= 0 => x = 0.
    
    # Ax = b => [1]*x = [-1]
    
    A = torch.tensor([[[1.0]]]) # (1, 1, 1)
    b = torch.tensor([[-1.0]])   # (1, 1)
    
    system = LinearSystemPair([A], [b])
    
    solver = LBFGSBSolver(system, output_dir=tmp_path)
    x0 = torch.zeros(1, 1, 1)
    x_est = solver.solve(x0, n_iter=20, verbose=False)
    
    print(f"Estimated x: {x_est}")
    assert x_est.item() >= 0.0
    assert torch.abs(x_est).item() < 1e-6 # Should be 0

def test_lbfgsb_random_large(tmp_path: Path):
    # Larger random problem
    torch.manual_seed(42)
    X, Y, Z = 5, 5, 10
    
    x_true = torch.rand(X, Y, Z) * (torch.rand(X, Y, Z) > 0.8).float() # Sparse x
    
    A1 = torch.randn(X, Y, Z)
    A2 = torch.randn(X, Y, Z)
    
    b1 = torch.sum(A1 * x_true, dim=-1).T
    b2 = torch.sum(A2 * x_true, dim=-1).T
    
    system = LinearSystemPair([A1, A2], [b1, b2])
    
    solver = LBFGSBSolver(system, output_dir=tmp_path)
    
    x0 = torch.zeros(X, Y, Z)
    x_est = solver.solve(x0, n_iter=50, verbose=True) # L-BFGS-B usually converges fast
    
    pred = system.forward(x_est)
    loss = torch.norm(pred - system.b)
    
    # Initial residual
    pred0 = system.forward(x0)
    loss0 = torch.norm(pred0 - system.b)
    print(f"Initial residual norm: {loss0.item()}")
    print(f"Final residual norm: {loss.item()}")
    
    assert loss < loss0 * 0.1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_lbfgsb_convergence_tiny(Path(tmp))
        test_lbfgsb_non_negativity(Path(tmp))
        test_lbfgsb_random_large(Path(tmp))
        print("All manual tests passed.")
