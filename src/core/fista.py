import torch
import logging
from typing import List, Tuple, Optional, Dict
from src.core.linear_system_pair import LinearSystemPair
from pathlib import Path
from src.core.solver import Solver

logger = logging.getLogger(__name__)

class FISTASolver(Solver):
    """
    FISTA solver for the problem:
    min_x 1/2 ||Ax - b||^2 + lambda * ||x||_1 + I_{x>=0}
    """
    def __init__(self, 
                 linear_system: LinearSystemPair,
                 output_dir: Path,
                 tag: str = "",
                 lambda_reg: float = 0.0, 
                 lipchitz: Optional[float] = None,
                 backtracking: bool = False, # Do not use backtracking to set lipschitz steps 1/L.
                 bt_eta: float = 2.0,
                 bt_max_steps: int = 20,
                 L_min: float = 1e-12,
                 L_max: float = 1e12):
        """
        Args:
            linear_system: An object with forward(x) and adjoint(residual) methods.
                           Can be a simple matrix or a complex operator.
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
        self.t = 1.0
        self.y = None
        # Workspace buffer to avoid allocating large temporaries each iteration.
        self._tmp = None
        self._grad = None

    def _pre_solve(self, x0: torch.Tensor) -> torch.Tensor:
        # Avoid cloning x0 in Solver._pre_solve; return a working tensor reference.
        # We still need y to be independent from x for the momentum update.
        x = super()._pre_solve(x0)
        self.y = x.clone().to(self.system.device)
        self._tmp = torch.empty_like(self.y)
        if self.L is None:
            # With backtracking enabled, we can start from a naive guess and adapt.
            # Start with something small but non-zero.
            self.L = 1.0
            logger.info(
                "Lipschitz constant not provided. Starting from L=1.0 and using backtracking=%s.",
                self.backtracking,
            )
        else:
            logger.info("Using provided Lipschitz constant L=%g (backtracking=%s)", self.L, self.backtracking)

        # Reset momentum (in case solver object is reused)
        self.t = 1.0
        return x

    def _objective(self, x: torch.Tensor) -> float:
        """Full objective: 0.5||Ax-b||^2 + lambda||x||_1 with x already constrained if desired."""
        Ax = self.system.forward(x)
        resid = Ax - self.system.b
        data_term = 0.5 * torch.sum(resid.float() ** 2)
        reg_term = self.lambda_reg * torch.sum(torch.abs(x.float()))
        return float((data_term + reg_term).item())

    def _smooth_value_and_grad(self, y: torch.Tensor):
        """Return f(y)=0.5||Ay-b||^2 and grad f(y)=A^T(Ay-b)."""
        Ay = self.system.forward(y)
        resid = Ay - self.system.b
        f_y = 0.5 * torch.sum(resid.float() ** 2)
        grad = self.system.adjoint(resid)
        return f_y, grad

    def _prox(self, v: torch.Tensor, step_size: float) -> torch.Tensor:
        """prox_{lambda/L ||.||_1 + I_{x>=0}}(v)."""
        threshold = self.lambda_reg * step_size
        # In-place optimization: v = max(0, v - threshold)
        if threshold > 0:
            v.sub_(threshold)
        v.clamp_(min=0) # Non negative assumption to avoid negative values
        return v

    def _prox_from_y_grad_(self, step_size: float) -> torch.Tensor:
        """Compute x = prox(y - step_size*grad) using reusable buffers.

        This avoids materializing the huge temporary `(y - step_size * grad)`
        each iteration, which otherwise triggers a large allocation and can OOM.

        Requires `self._tmp` to be allocated and `self._grad` to be set.
        Returns a new tensor `x_next` (because we must keep `y` intact for momentum).
        """
        # tmp = y - step*grad   (in-place into tmp)
        self._tmp.copy_(self.y)
        self._tmp.add_(self._grad, alpha=-step_size)

        # x_next = prox(tmp) (prox is in-place, so clone first)
        x_next = self._tmp.clone()
        return self._prox(x_next, step_size)
        
    def _solve_step(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Runs one step of the FISTA algorithm.
        """
        if k == 0:
            self._log_memory(f"start of step {k}") # memory foot print same each iterations
        
        L = float(self.L)
        L = max(self.L_min, min(self.L_max, L))

        # Compute smooth value and gradient at y
        f_y, grad = self._smooth_value_and_grad(self.y)
        # Keep gradient for reuse in candidate construction without extra temporaries.
        self._grad = grad
        
        if k == 0:
            self._log_memory(f"step {k}, after smooth grad")

        # Backtracking line search (Beck & Teboulle): find L such that
        # f(x) <= f(y) + <grad, x-y> + (L/2)||x-y||^2
        if self.backtracking:
            bt_ok = False
            for bt in range(self.bt_max_steps):
                step_size = 1.0 / L
                x_candidate = self._prox_from_y_grad_(step_size)

                # Majorization check for smooth part f
                # f(x_candidate)
                Ax_c = self.system.forward(x_candidate)
                resid_c = Ax_c - self.system.b
                f_c = 0.5 * torch.sum(resid_c.float() ** 2)

                diff = x_candidate - self.y
                q = f_y + torch.sum(grad.float() * diff.float()) + 0.5 * L * torch.sum(diff.float() ** 2)

                if f_c <= q + 1e-12:
                    bt_ok = True
                    x_next = x_candidate
                    break

                L = min(self.L_max, L * self.bt_eta)

            if not bt_ok:
                # Fall back: accept last candidate even if check failed, but warn.
                logger.warning(
                    "Backtracking did not satisfy condition after %d steps at iter %d; "
                    "using L=%g. Consider increasing bt_max_steps or bt_eta.",
                    self.bt_max_steps,
                    k,
                    L,
                )
                step_size = 1.0 / L
                x_next = self._prox_from_y_grad_(step_size)
        else:
            step_size = 1.0 / L
            x_next = self._prox_from_y_grad_(step_size)

        self.L = L
        
        # 3. FISTA Momentum update
        t_next = (1 + (1 + 4 * self.t**2)**0.5) / 2
        beta = (self.t - 1) / t_next
        self.y.zero_().add_(x_next, alpha=1 + beta).add_(x, alpha=-beta)
        
        self.t = t_next
        if k == 0:
            self._log_memory(f"end of step {k}")
        return x_next

    def _log_memory(self, step_name: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(f"Memory after '{step_name}': {allocated:.2f} MB (peak: {max_allocated:.2f} MB)")

    def soft_threshold(self, x, threshold):
        return torch.nn.functional.softshrink(x, threshold)

    def project_non_negative(self, x):
        return torch.clamp(x, min=0)

    def log(self, x, k):
        # Use the base-class logging implementation, which moves stats to CPU and
        # reports timing. This avoids extra GPU allocations during logging.
        super().log(x, k)