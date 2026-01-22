import torch
import logging
from typing import Optional
from src.core.linear_system_pair import LinearSystemPair
from pathlib import Path
from src.core.solver import Solver

logger = logging.getLogger(__name__)

class ISTASolver(Solver):
    """
    ISTA solver for the problem:
    min_x 1/2 ||Ax - b||^2 + lambda * ||x||_1 + I_{x>=0}
    """
    def __init__(self, 
                 linear_system: LinearSystemPair,
                 output_dir: Path,
                 tag: str = "",
                 lambda_reg: float = 0.0, 
                 lipchitz: Optional[float] = None,
                 backtracking: bool = False,
                 bt_eta: float = 2.0,
                 bt_max_steps: int = 20,
                 L_min: float = 1e-12,
                 L_max: float = 1e12):
        """
        Args:
            linear_system: An object with forward(x) and adjoint(residual) methods.
            lambda_reg: Regularization parameter for L1.
            lipchitz: Lipschitz constant (max eigenvalue of A^T A). If None, estimated.
        """
        super().__init__(linear_system, output_dir, tag)
        self.lambda_reg = lambda_reg
        self.L = lipchitz
        self.backtracking = backtracking
        self.bt_eta = bt_eta
        self.bt_max_steps = bt_max_steps
        self.L_min = L_min
        self.L_max = L_max

    def _pre_solve(self, x0: torch.Tensor) -> torch.Tensor:
        x = super()._pre_solve(x0)
        if self.L is None:
            self.L = 1.0
            logger.info(
                "Lipschitz constant not provided. Starting from L=1.0 and using backtracking=%s.",
                self.backtracking,
            )
        else:
            logger.info("Using provided Lipschitz constant L=%g (backtracking=%s)", self.L, self.backtracking)
        return x

    def _prox(self, v: torch.Tensor, step_size: float) -> torch.Tensor:
        """prox_{lambda/L ||.||_1 + I_{x>=0}}(v)."""
        threshold = self.lambda_reg * step_size
        # In-place optimization: v = max(0, v - threshold)
        if threshold > 0:
            v.sub_(threshold)
        v.clamp_(min=0) # Non negative assumption
        return v

    def _solve_step(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Runs one step of the ISTA algorithm.
        """
        L = float(self.L)
        L = max(self.L_min, min(self.L_max, L))

        # 1. Compute Gradient at x
        # grad = A^T(Ax - b)
        Ax = self.system.forward(x)
        resid = Ax - self.system.b
        grad = self.system.adjoint(resid)
        
        # Current function value (smooth part)
        f_x = 0.5 * torch.sum(resid.float() ** 2)

        # 2. Backtracking line search
        if self.backtracking:
            bt_ok = False
            for bt in range(self.bt_max_steps):
                step_size = 1.0 / L
                
                # x_next = prox(x - step * grad)
                # We clone x first to avoid modifying it, then subtract grad
                # Note: For efficiency one might use buffers, but for now we prioritize clarity/correctness
                x_candidate = x - step_size * grad
                x_candidate = self._prox(x_candidate, step_size)

                # Check majorization condition
                # f(x_candidate) <= Q_L(x_candidate, x)
                # Q_L(z, x) = f(x) + <grad, z-x> + L/2 ||z-x||^2
                
                Ax_c = self.system.forward(x_candidate)
                resid_c = Ax_c - self.system.b
                f_c = 0.5 * torch.sum(resid_c.float() ** 2)

                diff = x_candidate - x
                q = f_x + torch.sum(grad.float() * diff.float()) + 0.5 * L * torch.sum(diff.float() ** 2)

                if f_c <= q + 1e-12:
                    bt_ok = True
                    x_next = x_candidate
                    break

                L = min(self.L_max, L * self.bt_eta)

            if not bt_ok:
                logger.warning(
                    "Backtracking did not satisfy condition after %d steps at iter %d; "
                    "using L=%g.",
                    self.bt_max_steps,
                    k,
                    L,
                )
                step_size = 1.0 / L
                x_next = self._prox(x - step_size * grad, step_size)
        else:
            step_size = 1.0 / L
            x_next = self._prox(x - step_size * grad, step_size)

        self.L = L
        
        # ISTA: No momentum update
        return x_next
