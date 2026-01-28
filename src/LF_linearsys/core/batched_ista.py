import torch
import logging
import time
from pathlib import Path
from LF_linearsys.core.point_system import PointLinearSystem

logger = logging.getLogger(__name__)

class BatchedISTASolver:
    """
    Batched ISTA solver.
    Solves min_x 0.5 * ||Ax - b||^2 + lambda * ||x||_1
    Constraint: x >= 0 (optional, but usually desired for light fields)
    """
    def __init__(self, 
                 system: PointLinearSystem,
                 lambda_reg: float = 0.0,
                 n_iter: int = 100,
                 output_dir: Path = None,
                 positivity: bool = True):
        self.system = system
        self.lambda_reg = lambda_reg
        self.n_iter = n_iter
        self.output_dir = output_dir
        self.positivity = positivity
        
        # Estimate Lipschitz
        logger.info("Estimating Lipschitz constants...")
        self.L = self.system.estimate_lipschitz(num_iters=20)
        # Safety margin
        self.L = self.L * 1.1
        self.step_size = 1.0 / self.L
        
        # Handle zeros in L (if A is all zero for some points)
        # Avoid division by zero
        self.step_size[self.L < 1e-8] = 0.0
        
        # Reshape step_size to (B, 1) for broadcasting
        self.step_size = self.step_size.unsqueeze(1)
        
    def solve(self, x0: torch.Tensor = None) -> torch.Tensor:
        """
        Runs Batched ISTA.
        """
        B, N = self.system.B, self.system.N
        
        if x0 is None:
            x = torch.zeros(B, N, device=self.system.device)
        else:
            x = x0.clone()
            
        logger.info(f"Starting Batched ISTA for {B} points over {self.n_iter} iterations...")
        start_time = time.time()
        
        for k in range(self.n_iter):
            # 1. Gradient step
            # grad = A^T (A x - b)
            Ax = self.system.forward(x)
            resid = Ax - self.system.b
            grad = self.system.adjoint(resid)
            
            # 2. Descent
            # x_{k+1} = prox(x - step * grad)
            x_next = x - self.step_size * grad
            
            # 3. Proximal operator (Soft Thresholding + Non-negativity)
            # Threshold = lambda * step_size
            threshold = self.lambda_reg * self.step_size
            
            # Soft thresholding: sign(x) * max(|x| - thresh, 0)
            # With positivity constraint: max(x - thresh, 0)
            if self.positivity:
                 x_next = torch.clamp(x_next - threshold, min=0.0)
            else:
                 x_next = torch.sign(x_next) * torch.clamp(torch.abs(x_next) - threshold, min=0.0)
                 
            x = x_next
            
            if k % 10 == 0 or k == self.n_iter - 1:
                # Log stats (subset to avoid huge overhead)
                with torch.no_grad():
                    # Just mean loss over batch
                    # loss = 0.5 ||Ax-b||^2 + reg
                    Ax = self.system.forward(x)
                    r = Ax - self.system.b
                    fit = 0.5 * torch.sum(r**2) / B
                    reg = self.lambda_reg * torch.norm(x, p=1) / B
                    obj = fit + reg
                    logger.info(f"Iter {k}: Mean Obj={obj:.6f}, Mean Fit={fit:.6f}")
                    
        total_time = time.time() - start_time
        logger.info(f"Solved {B} points in {total_time:.2f}s ({(total_time/B)*1000:.2f} ms/point)")
        
        return x
