import torch
import logging
import os
from pathlib import Path
from LF_linearsys.core.point_system import PointLinearSystem

logger = logging.getLogger(__name__)

class BatchedRegNewtonLSSolver:
    """Projected Regularized Newton Solver for Batched Point Linear Systems.
    Minimizes:
        f(x) = 0.5 * ||Ax - b||^2 + 0.5 * lambda_reg * ||Dx||^2
    subject to x >= 0.

    Update Rule:
        H_k \Delta x = - \nabla f(x_k)
        x_{k+1} = max(0, x_k + alpha * \Delta x)
    
    Where:
        \nabla f(x) = A^T(Ax - b) + lambda_reg * D^T D x
        H = A^T A + lambda_reg * D^T D
    """
    def __init__(self, system: PointLinearSystem, lambda_reg: float = 0.0, n_iter: int = 20, output_dir: Path = None, positivity: bool = True, **kwargs):
        self.system = system
        self.lambda_reg = lambda_reg
        self.n_iter = n_iter
        self.output_dir = output_dir
        self.positivity = positivity
        
        # Dimensions
        self.B = system.B
        self.M = system.M
        self.N = system.N # This is Z (depth)
        
        self.device = system.device
        
        # Precompute D (Finite Difference Matrix)
        # D is (N-1, N) or (N, N). Let's use (N, N) for simplicity (circular or zero pad)
        # Or more typically (N-1, N).
        # Let's construct D s.t. Dx computes x[i+1] - x[i].
        # D shape (N-1, N).
        if self.N > 1:
            D = torch.zeros(self.N - 1, self.N, device=self.device)
            # D[i, i] = -1, D[i, i+1] = 1
            idx = torch.arange(self.N - 1, device=self.device)
            D[idx, idx] = -1.0
            D[idx, idx + 1] = 1.0
            self.D = D
        else:
            self.D = torch.zeros(1, 1, device=self.device)
            
        # Precompute D^T D which is constant (N, N)
        self.DtD = torch.matmul(self.D.T, self.D) # (N, N)
        
        # Precompute A^T A for each batch (or check if we can compute it efficiently)
        # A: (B, M, N)
        # AtA: (B, N, N)
        logger.info("Precomputing A^T A for Newton Solver...")
        self.AtA = torch.bmm(self.system.At, self.system.A) # (B, N, N)
        
        # Hessians
        # H = AtA + lambda * DtD
        # We can precompute the symetric constant part of Hessian if lambda is fixed.
        # using fix inverse of hessian then projecting the result. This is a specific variant (often called Projected Newton with fixed Hessian).
        
        self.H = self.AtA + self.lambda_reg * self.DtD.unsqueeze(0) # Broadcast (B, N, N)

        # Since H is constant and SPD (if A is full rank or lambda > 0), we can Cholesky it once.
        # Factorize H for fast solving. Laplacian DtD is semi-definite and do not guarantee strong convexity.
        # Add epsilon to diagonal for stability
        self.H += 1e-5 * torch.eye(self.N, device=self.device).unsqueeze(0)
        
        try:
            self.L_chol = torch.linalg.cholesky(self.H)
            self.use_cholesky = True
        except RuntimeError:
            logger.warning("Cholesky decomposition failed (Matrix not PD). Falling back to solve.")
            self.use_cholesky = False

    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-batch objective value.

        f(x) = 0.5 * ||Ax - b||^2 + 0.5 * lambda_reg * ||Dx||^2
        Returns: (B,) tensor.
        """
        Ax = self.system.forward(x)
        res = Ax - self.system.b
        loss_data = 0.5 * torch.sum(res ** 2, dim=1)

        Dx = torch.matmul(x, self.D.T)
        loss_reg = 0.5 * self.lambda_reg * torch.sum(Dx ** 2, dim=1)
        return loss_data + loss_reg
            
    def solve(self, x0=None):
        if x0 is None:
            # Initialize with zeros or ATb
            # x0 = torch.zeros(self.B, self.N, device=self.device)
            x0 = torch.zeros(self.B, self.N, device=self.device)
            
        x = x0
        
        # Step size alpha. 
        # For pure quadratic problems, Newton step size is 1.0.
        # With projection, we might need backtracking, but let's start with 1.0 or 0.5 dampening.
        alpha = 1.0 
        
        for k in range(self.n_iter):
            # 1. Compute Gradient
            # grad = A^T(Ax - b) + lambda D^T D x
            Ax = self.system.forward(x)
            residual = Ax - self.system.b
            grad_data = self.system.adjoint(residual) # (B, N) 
            
            grad_reg = self.lambda_reg * torch.matmul(x, self.DtD.T) # x is (B,N), DtD is (N,N). x @ DtD = (DtD x)^T
            
            grad = grad_data + grad_reg
            
            # 2. Solve Newton System
            # H \Delta x = -grad
            # \Delta x = - H^{-1} grad
            
            # grad is (B, N). unsqueeze to (B, N, 1)
            rhs = -grad.unsqueeze(2)
            
            if self.use_cholesky:
                # cholesky_solve expects (B, N, 1) and L (B, N, N)
                delta_x = torch.cholesky_solve(rhs, self.L_chol).squeeze(2)
            else:
                delta_x = torch.linalg.solve(self.H, rhs).squeeze(2)
                
            # 3. Update and Project
            x_new = x + alpha * delta_x
            
            if self.positivity:
                x_new = torch.clamp(x_new, min=0.0)
            
            # Check convergence
            diff = torch.norm(x_new - x) / (torch.norm(x) + 1e-9)
            x = x_new
            
            if diff < 1e-7:
                logger.info(f"Converged at iteration {k} with diff {diff:.2e}")
                break
            
            if k % 10 == 0:
                # Log residual
                Ax_new = self.system.forward(x)
                r_norm = torch.norm(Ax_new - self.system.b)
                logger.info(f"Iter {k}: ||Ax-b|| = {r_norm.item():.4e}, diff = {diff.item():.2e}")
                
        return x