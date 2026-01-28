import torch
import pytest
import logging
from LF_linearsys.core.point_system import PointLinearSystem
from LF_linearsys.core.batched_newton_linesearch import BatchedRegNewtonLSSolver
from LF_linearsys.core.batched_newton_activeset import BatchedRegNewtonASSolver

def test_batched_newton_solver():
    """
    Test the Projected Regularized Newton Solver on a small synthetic batch.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 1. Setup Data
    # Batch size B=2, Measurements M=5, Depth Z=10
    B, M, Z = 2, 5, 10
    device = 'cpu'
    
    # Random A
    torch.manual_seed(42)
    A = torch.randn(B, M, Z, device=device)
    
    # True x (sparse)
    x_true = torch.zeros(B, Z, device=device)
    x_true[0, 2] = 1.0
    x_true[0, 5] = 0.5
    x_true[1, 3] = 0.8
    x_true[1, 7] = 0.2
    
    # b = Ax
    # (B, M, Z) @ (B, Z, 1) -> (B, M, 1) -> (B, M)
    b = torch.bmm(A, x_true.unsqueeze(2)).squeeze(2)
    
    # 2. Create System
    system = PointLinearSystem(A, b, device=device)
    
    # 3. Solve
    # Use small lambda to allow fitting data well
    solver = BatchedRegNewtonLSSolver(system, lambda_reg=0.01, n_iter=20, positivity=True)
    
    # Test _compute_loss
    loss_0 = solver._compute_loss(torch.zeros_like(x_true))
    # loss should be 0.5 * ||b||^2
    expected_loss = 0.5 * torch.sum(b**2, dim=1)
    assert torch.allclose(loss_0, expected_loss, atol=1e-5)
    
    x_hat = solver.solve()
    
    # 4. Check results
    assert x_hat.shape == (B, Z)
    assert torch.all(x_hat >= 0)
    
    # Check error
    # Newton with L2 reg should come close to L2 solution
    # Note: L2 regularization on finite differences (smoothness) is different from true x (sparse)
    # So we don't expect exact reconstruction of x_true, but we expect low residual on Ax-b
    
    Ax_hat = system.forward(x_hat)
    residual_norm = torch.norm(Ax_hat - b, dim=1)
    
    logger.info(f"Residual norms: {residual_norm}")
    
    # Expect residual to be reasonably small (it's solving min ||Ax-b||^2 + ...)
    assert torch.all(residual_norm < 1.0)
    
    # Check if smoothness is working (optional/qualitative)
    # D x should be small
    D = solver.D
    Dx = torch.matmul(x_hat, D.T)
    Dx_norm = torch.norm(Dx, dim=1)
    logger.info(f"Smoothness norms (||Dx||): {Dx_norm}")

def test_active_set_newton_solver():
    """
    Test the Active Set Newton Solver on a small synthetic batch.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 1. Setup Data
    B, M, Z = 2, 5, 10
    device = 'cpu'
    
    torch.manual_seed(42)
    A = torch.randn(B, M, Z, device=device)
    
    # True x (sparse)
    x_true = torch.zeros(B, Z, device=device)
    x_true[0, 2] = 1.0
    x_true[0, 5] = 0.5
    x_true[1, 3] = 0.8
    x_true[1, 7] = 0.2
    
    b = torch.bmm(A, x_true.unsqueeze(2)).squeeze(2)
    
    system = PointLinearSystem(A, b, device=device)
    
    # 3. Solve
    solver = BatchedRegNewtonASSolver(system, lambda_reg=0.01, n_iter=20, positivity=True)
    
    loss_0 = solver._compute_loss(torch.zeros_like(x_true))
    expected_loss = 0.5 * torch.sum(b**2, dim=1)
    assert torch.allclose(loss_0, expected_loss, atol=1e-5)
    
    x_hat = solver.solve()
    
    # 4. Check results
    assert x_hat.shape == (B, Z)
    assert torch.all(x_hat >= 0)
    
    Ax_hat = system.forward(x_hat)
    residual_norm = torch.norm(Ax_hat - b, dim=1)
    
    logger.info(f"AS-Newton Residual norms: {residual_norm}")
    assert torch.all(residual_norm < 1.0)

if __name__ == "__main__":
    test_batched_newton_solver()
    test_active_set_newton_solver()
