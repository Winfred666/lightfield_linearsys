import torch
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from LF_linearsys.core.point_system import PointLinearSystem

logger = logging.getLogger(__name__)

class BatchedSVDSolver:
    """
    Direct SVD-based Tikhonov Regularized Solver for Batched Point Linear Systems.
    Minimizes:
        f(x) = 0.5 * ||Ax - b||^2 + 0.5 * lambda_reg * ||x||^2
    
    This solver provides an analytical solution via SVD and does not explicitly 
    enforce positivity constraints.
    """
    def __init__(self, system: PointLinearSystem, lambda_reg: float = 0.0, n_iter: int = 1, output_dir: Path = None, **kwargs):
        self.system = system
        self.lambda_reg = lambda_reg
        self.n_iter = n_iter # For compatibility, usually 1 is enough for linear SVD
        self.output_dir = output_dir
        
        self.B = system.B
        self.M = system.M
        self.N = system.N
        self.device = system.device
        
        self.history: dict[str, list[float]] = {
            "iter": [],
            "residual_norm": [],
        }
        self.log_interval = int(kwargs.get("log_interval", 1))

    def _residual_norm(self, x: torch.Tensor) -> float:
        """Compute global residual norm ||Ax-b|| for the current batch (single scalar)."""
        with torch.no_grad():
            r = self.system.forward(x) - self.system.b
            return float(torch.linalg.vector_norm(r.detach().float()).item())

    def _maybe_save_residual_history(self, *, tag: str | None) -> None:
        """Optionally write residual history to disk.

        We keep this lightweight: for SVD solver we typically have only
        the initial and final residual.
        """
        if tag is None or self.output_dir is None:
            return
        if not self.history.get("residual_norm"):
            return

        out_dir = Path(self.output_dir) / "loss_curve"
        out_dir.mkdir(parents=True, exist_ok=True)
        residual_path = out_dir / f"residual_norm_{tag}.txt"
        with open(residual_path, "w") as f:
            for res in self.history["residual_norm"]:
                f.write(f"{res}\n")
        logger.info("Saved residual norm list to %s", residual_path)

    def _compute_loss(self, x):
        Ax = self.system.forward(x)
        res = Ax - self.system.b
        loss_data = 0.5 * torch.sum(res ** 2, dim=1)
        loss_reg = 0.5 * self.lambda_reg * torch.sum(x ** 2, dim=1)
        return loss_data + loss_reg

    def solve(self, x0=None, *, tag: str | None = None):
        record_history = tag is not None
        self.history["iter"] = []
        self.history["residual_norm"] = []

        x = torch.zeros(self.B, self.N, device=self.device) if x0 is None else x0.to(self.device)

        if record_history:
            # Keep SVD solver history minimal: just initial and final residuals.
            self.history["iter"].append(0)
            self.history["residual_norm"].append(self._residual_norm(x))
        
        # In a purely linear system, one SVD step is the solution.
        # for k in range(self.n_iter):
        # 1. Get Jacobian (A) and Residual (Ax - b)
        J = self.system.A # (B, M, N)
        Ax = self.system.forward(x)
        r = Ax - self.system.b # (B, M)

        # 2. SVD Decomposition
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)

        # 3. Apply Tikhonov Regularization analytically
        # Filter: sigma / (sigma^2 + lambda)
        S_inv_damped = S / (S**2 + self.lambda_reg + 1e-12)
        
        # 4. Truncate
        threshold = S.max(dim=1, keepdim=True).values * 1e-5
        S_inv_damped[S < threshold] = 0.0

        # 5. Compute Step: dx = V * S_inv * U^T * r
        # U^T r
        Ut_r = torch.matmul(U.transpose(1, 2), r.unsqueeze(2)) # (B, N, 1)
        
        # Scale by singular values
        scaled_Ut_r = Ut_r * S_inv_damped.unsqueeze(2) # (B, N, 1)
        
        # Project back to variable space (V is Vh.H)
        dx = torch.matmul(Vh.transpose(1, 2), scaled_Ut_r).squeeze(2) # (B, N)

        # 6. Update
        x = x - dx
        
        # if k % 5 == 0 or k == self.n_iter - 1:
        logger.info(f"SVD Solver: Residual Norm={torch.linalg.vector_norm(r).item():.4e}")

        if record_history:
            self.history["iter"].append(1)
            self.history["residual_norm"].append(self._residual_norm(x))
            self._maybe_save_residual_history(tag=tag)

        return x
