import torch
import logging
import numpy as np
import scipy.optimize
from typing import Optional
from LF_linearsys.core.linear_system_pair import LinearSystemPair
from pathlib import Path
from LF_linearsys.core.solver import Solver

logger = logging.getLogger(__name__)

class LBFGSBSolver(Solver):
    """
    L-BFGS-B solver for the problem:
    min_x 0.5 * ||Ax - b||^2  s.t. x >= 0
    
    Uses scipy.optimize.minimize(method='L-BFGS-B').
    Note: This moves the current estimate to CPU for each step, which may be slow
    for extremely large variables if the transfer time dominates the operator time.
    However, L-BFGS-B often requires fewer iterations than first-order methods like ISTA.
    """
    def __init__(self, 
                 linear_system: LinearSystemPair,
                 output_dir: Path,
                 tag: str = "",
                 lambda_reg: float = 0.0, 
                 **kwargs):
        """
        Args:
            linear_system: An object with forward(x) and adjoint(residual) methods.
            lambda_reg: L1 regularization weight. 
                        WARNING: Standard L-BFGS-B solves smooth problems. 
                        If lambda_reg > 0, this solver currently IGNORES it or 
                        treats it as 0, effectively solving the LS problem only.
        """
        super().__init__(linear_system, output_dir, tag)
        self.lambda_reg = lambda_reg
        if self.lambda_reg > 0:
            logger.warning("L-BFGS-B solver received lambda_reg=%g but L1 regularization is "
                           "not supported by smooth L-BFGS-B. Solving ||Ax-b||^2 only.", self.lambda_reg)

        self._current_iter = 0

    def _solve_step(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # Not used because we override solve()
        pass

    def solve(self, x0: torch.Tensor, n_iter: int = 100, verbose: bool = True) -> torch.Tensor:
        x = self._pre_solve(x0)
        
        logger.info(f"Starting L-BFGS-B solver (max_iter={n_iter})")

        # Prepare for scipy
        # Flatten and convert to numpy float64 (or float32)
        # We use float64 for stability in L-BFGS, even if GPU ops are float32/16.
        x_shape = x.shape
        x_np = x.detach().cpu().numpy().flatten().astype(np.float64)
        
        # Reset iteration counter for logging
        self._current_iter = 0
        
        # Define objective and gradient for scipy
        def func_and_grad(x_vec):
            # Convert back to torch tensor on device
            x_tensor = torch.from_numpy(x_vec).view(x_shape).to(self.system.device)
            if x_tensor.dtype != torch.float32:
                x_tensor = x_tensor.float() # Ensure float32 for compute

            # Forward
            Ax = self.system.forward(x_tensor)
            resid = Ax - self.system.b
            
            # Loss: 0.5 * ||Ax - b||^2
            # Note: resid might be mixed precision, ensure float for accumulation
            f_val = 0.5 * torch.sum(resid.float() ** 2).item()
            
            # Gradient: A^T (Ax - b)
            grad_tensor = self.system.adjoint(resid)
            
            # Convert grad to numpy
            grad_np = grad_tensor.detach().cpu().numpy().flatten().astype(np.float64)
            
            return f_val, grad_np

        # Callback for logging
        def callback(x_vec):
            if verbose and (self._current_iter % self.log_interval == 0 or self._current_iter == n_iter - 1):
                # Reconstruct tensor for logging (cheap if we just want stats, but log computes Ax again...)
                # Solver.log() calls forward(), so this adds cost.
                # Only do it if strictly needed.
                with torch.no_grad():
                    x_tensor = torch.from_numpy(x_vec).view(x_shape).to(self.system.device)
                    self.log(x_tensor, self._current_iter)
            
            self._current_iter += 1

        # Bounds: x >= 0
        # Use efficient Bounds object
        from scipy.optimize import Bounds
        bounds = Bounds(0, np.inf)

        # Run optimization
        # options: 'maxiter' is the max number of iterations.
        # 'disp': None
        res = scipy.optimize.minimize(
            func_and_grad,
            x_np,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                'maxiter': n_iter,
                'disp': False, # We handle logging
                'ftol': 1e-9,
                'gtol': 1e-9
            }
        )

        logger.info(f"L-BFGS-B finished: {res.message} (nit={res.nit})")
        
        # Final result
        x_final_np = res.x
        x_final = torch.from_numpy(x_final_np).view(x_shape).to(self.system.device).float()
        
        # Final log
        if verbose:
            self.log(x_final, self._current_iter)
            
        x_final = self._post_solve(x_final)
        return x_final
