import torch
import logging
import numpy as np
import time
from src.core.solver import Solver
from src.core.linear_system_pair import LinearSystemPair

logger = logging.getLogger(__name__)

class ProjectNewtonSolver(Solver):
    """
    Solver using Projected Newton method for Non-negative Least Squares.
    Solves min ||Ax - b||_2 subject to x >= 0.
    WARNING: for pair-wise solving or joint pair-wise solving, the matrix (BHW x WHD)
    1. Would not be block-diagonal because there are overlap of light field for differnt pairs.
    2. Might be extremely rank-deficient because only One measurement for certain (H,W, Z_reduced) area
    """
    def __init__(self, linear_system: LinearSystemPair, output_dir, tag="", **kwargs):
        super().__init__(linear_system, output_dir, tag=tag, **kwargs)

    def _solve_step(self, x: torch.Tensor, k: int) -> torch.Tensor:
        raise NotImplementedError("Projected Newton step not implemented yet.")

    def solve(self, x0: torch.Tensor, n_iter: int = 100, verbose: bool = True) -> torch.Tensor:
        raise NotImplementedError("Projected Newton solver not implemented yet.")