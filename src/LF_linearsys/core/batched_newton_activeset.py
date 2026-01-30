import torch
import logging
import os
from pathlib import Path
from LF_linearsys.core.point_system import PointLinearSystem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)

class BatchedRegNewtonASSolver:
    """
    Projected Regularized Newton Solver for Batched Point Linear Systems - Using Active Set.
    Minimizes:
        f(x) = 0.5 * ||Ax - b||^2 + 0.5 * lambda_reg * ||x||^2
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

        # Logging/history (similar spirit to core/Solver but simplified).
        self.log_interval = int(kwargs.get("log_interval", 5))
        self.history: dict[str, list[float] | list[int]] = {
            "iter": [],
            "loss": [],
            "residual_norm": [],
        }

        # Auto-tag each call to solve() so repeated runs don't overwrite artifacts.
        # (e.g. in unit tests where multiple solves happen in one output_dir)
        self._solve_call_idx = 0
        
        # Dimensions
        self.B = system.B
        self.M = system.M
        self.N = system.N # This is Z (depth)
        
        self.device = system.device
            
        # Precompute A^T A for each batch
        logger.info("Precomputing A^T A for Active Set Newton Solver...")
        self.AtA = torch.bmm(self.system.At, self.system.A) # (B, N, N)
        
        # Base Hessian H = AtA + lambda * I
        # Regularization ensures PD if lambda > 0.
        I = torch.eye(self.N, device=self.device).unsqueeze(0) # (1, N, N)
        self.H_base = self.AtA + self.lambda_reg * I

        # Log condition number of base Hessian
        # self._log_condition_number()
        
    def _compute_loss(self, x):
        """
        Compute f(x) = 0.5 * ||Ax - b||^2 + 0.5 * lambda_reg * ||x||^2
        """
        Ax = self.system.forward(x)
        res = Ax - self.system.b
        loss_data = 0.5 * torch.sum(res ** 2, dim=1)
        
        loss_reg = 0.5 * self.lambda_reg * torch.sum(x ** 2, dim=1)
        return loss_data + loss_reg

    def solve(self, x0=None, *, tag: str | None = None):
        # Only record per-iteration history (and plot) when the caller requests it.
        # In large runs this avoids extra CPU work + memory growth, and lets callers
        # decide to plot only every N batches.
        record_history = tag is not None

        # Reset per-run history.
        # (We always clear to avoid mixing histories across multiple solve() calls.)
        self.history["iter"] = []
        self.history["loss"] = []
        self.history["residual_norm"] = []

        x = torch.zeros(self.B, self.N, device=self.device) if x0 is None else x0.to(self.device)
        
        beta = 0.5
        c = 1e-6
        max_ls_iter = 20
        
        for k in range(self.n_iter):
            # --- 1. Compute Unconstrained Gradient ---
            Ax = self.system.forward(x)
            residual = Ax - self.system.b
            
            grad_data = self.system.adjoint(residual)
            grad_reg = self.lambda_reg * x
            grad = grad_data + grad_reg # (B, N)
            
            # --- 2. Determine Batched Active Set ---
            # Active if x is approx 0 AND gradient is positive (trying to push x negative)
            # Use small epsilon for zero check
            active_mask = (x <= 1e-7) & (grad > 0)

            # Fix for singular columns (e.g. where A is 0 for valid Z slices but no signal)
            # If diagonal of H is ~0, the system is singular. Treat these as active (frozen).
            # H_base is (B, N, N)
            h_diag = self.H_base.diagonal(dim1=-2, dim2=-1) # (B, N)
            null_mask = h_diag.abs() < 1e-8
            active_mask = active_mask | null_mask
            
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

            # Record mean loss / mean residual for plotting.
            # (loss is per-batch already)
            if record_history and (k % self.log_interval == 0 or k == self.n_iter - 1):
                with torch.no_grad():
                    mean_loss = float(f_x.mean().item())
                    resid_norm = float(torch.linalg.vector_norm(residual.detach().float()).item())
                self.history["iter"].append(int(k))
                self.history["loss"].append(mean_loss)
                self.history["residual_norm"].append(resid_norm)
            
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
                
        # Post-solve: export artifacts (optional)
        if record_history and (self.output_dir is not None):
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._plot_history(out_dir, tag=tag)

        return x

    def _plot_history(self, output_dir: Path, *, tag: str):
        """Save a simple convergence plot (residual norm + loss) like core/Solver."""
        if not self.history.get("iter"):
            logger.warning("AS-Newton history is empty; skipping plot.")
            return

        sub_dir = Path(output_dir) / "loss_curve"
        sub_dir.mkdir(parents=True, exist_ok=True)
        save_path = sub_dir / f"loss_curve_{tag}.png"

        fig, ax1 = plt.subplots(figsize=(10, 6))
        iters = list(self.history["iter"])

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Residual Norm")
        ax1.plot(iters, self.history["residual_norm"], label="Residual Norm")
        ax1.set_yscale("log")

        # ax2 = ax1.twinx()
        # ax2.set_ylabel("Mean Loss")
        # ax2.plot(iters, self.history["loss"], color="tab:orange", label="Mean Loss")
        # ax2.set_yscale("log")
        
        fig.tight_layout()
        plt.title(f"AS-Newton Convergence ({tag})")
        plt.savefig(save_path)
        plt.close(fig)
        logger.info("Saved AS-Newton convergence plot to %s", save_path)



    def _log_condition_number(self):
        """
        Compute condition number of the base Hessian H = A^T A + lambda*I for each batch.
        And log the mean, max and min of the condition number in one batch
        """
        cond_numbers = []
        for b in range(self.B):
            H_b = self.H_base[b].cpu().numpy()
            eigvals = np.linalg.eigvalsh(H_b)
            cond_num = eigvals[-1] / eigvals[0]
            cond_numbers.append(cond_num)
        cond_numbers = np.array(cond_numbers)
        logger.info(
            "Hessian condition numbers -- Mean: %.2e, Min: %.2e, Max: %.2e",
            cond_numbers.mean(),
            cond_numbers.min(),
            cond_numbers.max(),
        )
        return cond_numbers

