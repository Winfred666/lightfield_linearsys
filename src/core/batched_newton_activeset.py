import torch
import logging
import os
from pathlib import Path
from src.core.point_system import PointLinearSystem

logger = logging.getLogger(__name__)

class BatchedRegNewtonASSolver:
    """
    Projected Regularized Newton Solver for Batched Point Linear Systems - Using Active Set.
    Minimizes:
        f(x) = 0.5 * ||Ax - b||^2 + 0.5 * lambda_reg * ||Dx||^2
    subject to x >= 0.

    Strategy:
        At each iteration:
        1. Identify active set: indices where x <= 0 and grad > 0.
        2. Modify Hessian: Zero out rows/cols for active indices, set diagonal to 1.
        3. Zero out gradient for active indices.
        4. Solve for search direction: H_mod * delta_x = -grad_mod.
        5. Perform projected line search.
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
        if self.N > 1:
            D = torch.zeros(self.N - 1, self.N, device=self.device)
            idx = torch.arange(self.N - 1, device=self.device)
            D[idx, idx] = -1.0
            D[idx, idx + 1] = 1.0
            self.D = D
        else:
            self.D = torch.zeros(1, 1, device=self.device)
            
        # Precompute D^T D which is constant (N, N)
        self.DtD = torch.matmul(self.D.T, self.D) # (N, N)
        
        # Precompute A^T A for each batch
        logger.info("Precomputing A^T A for Active Set Newton Solver...")
        self.AtA = torch.bmm(self.system.At, self.system.A) # (B, N, N)
        
        # Base Hessian H = AtA + lambda * DtD + eps * I
        self.H_base = self.AtA + self.lambda_reg * self.DtD.unsqueeze(0) # Broadcast (B, N, N)
        self.H_base += 1e-5 * torch.eye(self.N, device=self.device).unsqueeze(0)
        
    def _compute_loss(self, x):
        """
        Compute f(x) = 0.5 * ||Ax - b||^2 + 0.5 * lambda_reg * ||Dx||^2
        """
        Ax = self.system.forward(x)
        res = Ax - self.system.b
        loss_data = 0.5 * torch.sum(res ** 2, dim=1)
        
        Dx = torch.matmul(x, self.D.T)
        loss_reg = 0.5 * self.lambda_reg * torch.sum(Dx ** 2, dim=1)
        return loss_data + loss_reg

    def solve(self, x0=None):
        x = torch.zeros(self.B, self.N, device=self.device) if x0 is None else x0.to(self.device)
        
        beta = 0.5
        c = 1e-6
        max_ls_iter = 20
        
        for k in range(self.n_iter):
            # --- 1. Compute Unconstrained Gradient ---
            Ax = self.system.forward(x)
            residual = Ax - self.system.b
            
            grad_data = self.system.adjoint(residual)
            grad_reg = self.lambda_reg * torch.matmul(x, self.DtD.T)
            grad = grad_data + grad_reg # (B, N)
            
            # --- 2. Determine Batched Active Set ---
            # Active if x is approx 0 AND gradient is positive (trying to push x negative)
            # Use small epsilon for zero check
            active_mask = (x <= 1e-7) & (grad > 0)
            
            # --- 3. Build/Modify Batched Hessian ---
            # Start with base Hessian
            H_k = self.H_base.clone() # (B, N, N)
            
            # Modify Hessian if any variables are active
            if active_mask.any():
                # active_mask is (B, N)
                
                # Zero out rows for active vars: H[b, i, :] = 0
                row_mask = (~active_mask).float().unsqueeze(2) # (B, N, 1)
                H_k = H_k * row_mask
                
                # Zero out cols for active vars: H[b, :, i] = 0
                col_mask = (~active_mask).float().unsqueeze(1) # (B, 1, N)
                H_k = H_k * col_mask
                
                # Set diagonal to 1 for active vars: H[b, i, i] = 1
                # This ensures the linear system is solvable and delta_x[i] becomes 0 (since grad[i]=0)
                # Here we explicitly set equation 1 * delta_x[i] = 0.
                H_k.diagonal(dim1=-2, dim2=-1).add_(active_mask.float())
                
                # --- Modify Gradient ---
                # For active variables, we want delta_x = 0.
                # The equation is H_k * delta_x = -grad_mod.
                # Since row i of H_k is [0...0 1 0...0], we need row i of -grad_mod to be 0
                # so that 1 * delta_x[i] = 0.
                grad = grad * (~active_mask).float()
                
            # --- 4. Solve for Newton Direction ---
            # H_k is symmetric. We can try Cholesky or eigen linear solve.
            try:
                # Use cholesky if possible for speed, but fallback to solve/lstsq
                # H_k should be PD because principal submatrix of PD is PD, and we added Identity blocks.
                L = torch.linalg.cholesky(H_k)
                delta_x = torch.cholesky_solve(-grad.unsqueeze(2), L).squeeze(2)
            except RuntimeError:
                # Fallback
                delta_x = torch.linalg.solve(H_k, -grad.unsqueeze(2)).squeeze(2)
            
            # --- 5. Vectorized Projected Line Search ---
            f_x = self._compute_loss(x)
            
            # Per-batch step sizes (shape: (B,))
            alpha_vec = torch.full((self.B,), 0.5, device=self.device)
            search_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
            x_next = x.clone()
            
            for ls_i in range(max_ls_iter):
                x_cand = x + alpha_vec[:, None] * delta_x
                if self.positivity:
                    x_cand = torch.clamp(x_cand, min=0.0)
                
                d_step = x_cand - x
                f_cand = self._compute_loss(x_cand)
                # Standard Armijo uses original gradient for descent check on the original function f.
                # But our step delta_x is computed w.r.t modified subspace.
                # Let's use the actual function decrease check.
                dir_deriv = torch.sum(grad * d_step, dim=1) # grad is the modified one? Or original?
                
                # Strict Armijo: f(x+p) <= f(x) + c * p^T * grad_f(x)
                # grad with holes for active set variable still represent original gradient for those free variable.
                cond = f_cand <= (f_x + c * dir_deriv)
                
                newly_accepted = cond & search_mask # Only consider those still searching
                if newly_accepted.any():
                    x_next[newly_accepted] = x_cand[newly_accepted]
                    search_mask[newly_accepted] = False
                
                if not search_mask.any():
                    break
                
                alpha_vec[search_mask] *= beta
            
            # For batches that did not find a step: clean freeze (no unverified update)
            # Leave x_next equal to x for those batches.
            if search_mask.any() and k % 5 == 0:
                logger.warning(
                    "AS-Newton line search failed for %d/%d batches at iter %d; freezing those batches.",
                    int(search_mask.sum().item()),
                    self.B,
                    k,
                )

            d_total = x_next - x
            x = x_next
            
            # Convergence check
            mean_diff = (torch.norm(d_total, dim=1) / (torch.norm(x, dim=1) + 1e-9)).mean().item()
            
            if k % 5 == 0:
                mean_loss = f_x.mean().item()
                mean_alpha = alpha_vec.mean().item()
                n_active = active_mask.float().sum(dim=1).mean().item()
                logger.info(f"AS-Newton Iter {k}: Loss={mean_loss:.4e}, RelDiff={mean_diff:.2e}, MeanAlpha={mean_alpha:.2e}, AvgActive={n_active:.1f}")
                
            if mean_diff < 1e-6:
                logger.info(f"Converged at iter {k}")
                break
                
        return x
