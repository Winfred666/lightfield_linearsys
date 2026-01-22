# Technique for zero 

import logging
from dataclasses import dataclass
from typing import Optional, List

import torch

from src.core.linear_system_pair import LinearSystemPair

logger = logging.getLogger(__name__)


@dataclass
class MaskingStats:
    n_total_vars: int
    n_free_vars: int
    n_fixed_zero: int


class LinearSystemPairMasked(LinearSystemPair):
    r"""A subclass of :class:`~src.core.linear_system.LinearSystem` that removes
    variables that are provably zero under the non-negativity constraint.

    Motivation
    ----------
    In our reconstruction problems we always enforce $x \ge 0$.

    If for some measurement-row/pixel $i$ we have:
    - $b_i = 0$
    - and the corresponding row of A has some positive contribution for variable $j$

    then any feasible non-negative solution must satisfy $x_j = 0$ (because otherwise
    $\langle A_i, x\rangle > 0$).

    We exploit this by *fixing* those variables to zero and solving a reduced problem
    over the remaining free variables.

    Notes / assumptions
    -------------------
    This inference is only sound if the contributing A entries are non-negative.
    We therefore only infer zeros from entries with $A_{ij} > threshold_A$.

    Interface
    ---------
    - `forward(x_free)`: accepts reduced vector.
    - `adjoint(residual)`: returns reduced vector.
    - `pack_x(x_full)`: Maps full (X,Y,Z) tensor to reduced vector.
    - `expand_x(x_free)`: Maps reduced vector to full (X,Y,Z) tensor.
    """

    def __init__(
        self,
        A_list: List[torch.Tensor],
        b_list: List[torch.Tensor],
        device='cpu',
        threshold_A: float = 0.0,
        threshold_b: float = 0.0,
        assume_non_negative_A: bool = True,
    ):
        # Initialize the base LinearSystem (filters rows, sets up valid_A, valid_b, etc.)
        super().__init__(A_list, b_list, device=device, threshold_A=threshold_A, threshold_b=threshold_b)

        self.assume_non_negative_A = assume_non_negative_A
        self.threshold_b = float(threshold_b)
        self.threshold_A = float(threshold_A)

        self._build_masks()

    @property
    def stats(self) -> MaskingStats:
        return MaskingStats(
            n_total_vars=int(self.X * self.Y * self.Z),
            n_free_vars=int(self.free_mask.sum().item()),
            n_fixed_zero=int((~self.free_mask).sum().item()),
        )

    def _build_masks(self) -> None:
        # self.valid_A: (N_valid, Z)
        # self.valid_indices: (N_valid,) in [0, X*Y)
        # self.b: (N_valid,)

        if self.b.numel() == 0:
            # Degenerate case: no valid rows.
            self.fixed_zero_mask = torch.zeros(self.X * self.Y * self.Z, dtype=torch.bool, device=self.device)
            self.free_mask = ~self.fixed_zero_mask
            self._free_flat_indices = torch.nonzero(self.free_mask, as_tuple=False).reshape(-1)
            logger.warning("MaskedLinearSystem: base system has no rows; no variables will be masked.")
            return

        b = self.b
        # row mask: b == 0 (within threshold_A)
        if self.threshold_b <= 0:
            zero_b_rows = b == 0
        else:
            zero_b_rows = torch.abs(b.float()) <= self.threshold_b

        if not torch.any(zero_b_rows):
            self.fixed_zero_mask = torch.zeros(self.X * self.Y * self.Z, dtype=torch.bool, device=self.device)
            self.free_mask = ~self.fixed_zero_mask
            self._free_flat_indices = torch.nonzero(self.free_mask, as_tuple=False).reshape(-1)
            logger.info("MaskedLinearSystem: no b==0 rows; no variable masking applied.")
            return

        A = self.valid_A
        idx = self.valid_indices

        # Only use positive contributions for soundness.
        if self.assume_non_negative_A:
            pos = A > self.threshold_A
        else:
            # If A may be negative, we cannot safely infer x_j=0.
            pos = torch.zeros_like(A, dtype=torch.bool)

        # Candidate mask: for rows with b=0, any z with A>0 forces x[pixel,z]=0
        inferred_fixed = pos[zero_b_rows]
        if inferred_fixed.numel() == 0:
            self.fixed_zero_mask = torch.zeros(self.X * self.Y * self.Z, dtype=torch.bool, device=self.device)
        else:
            # Expand pixel indices to variable indices in flattened XYZ ordering.
            # Pixel order in LinearSystem flatten: x.reshape(-1, Z) where -1 index is pixel.
            # Variable flat index = pixel_index * Z + z
            pix = idx[zero_b_rows].to(torch.long)
            # inferred_fixed: (N_zero_rows, Z)
            z_ids = torch.arange(self.Z, device=self.device, dtype=torch.long).unsqueeze(0).expand(pix.numel(), -1)
            var_ids = pix.unsqueeze(1) * self.Z + z_ids

            fixed_var_ids = var_ids[inferred_fixed]
            self.fixed_zero_mask = torch.zeros(self.X * self.Y * self.Z, dtype=torch.bool, device=self.device)
            if fixed_var_ids.numel() > 0:
                self.fixed_zero_mask[fixed_var_ids] = True

        self.free_mask = ~self.fixed_zero_mask
        self._free_flat_indices = torch.nonzero(self.free_mask, as_tuple=False).reshape(-1)

        logger.info(
            "MaskedLinearSystem: inferred %d fixed-zero vars out of %d (free=%d) from %d/%d b==0 rows.",
            int(self.fixed_zero_mask.sum().item()),
            int(self.X * self.Y * self.Z),
            int(self.free_mask.sum().item()),
            int(zero_b_rows.sum().item()),
            int(b.numel()),
        )

    # Like a proxy that freeze and keep the certain zero in solution. 
    def pack_x(self, x_full: torch.Tensor) -> torch.Tensor:
        """Pack full x (X,Y,Z) into a 1D vector of free variables."""
        if x_full.shape != (self.X, self.Y, self.Z):
            raise ValueError(f"x_full must have shape {(self.X, self.Y, self.Z)}, got {tuple(x_full.shape)}")
        x_flat = x_full.reshape(-1)
        return x_flat[self._free_flat_indices]

    def expand_x(self, x_free: torch.Tensor, *, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Expand free variable vector back to full x (X,Y,Z), filling fixed vars with 0."""
        x_free = x_free.reshape(-1)
        # Use dtype/device from self.valid_A if like is not provided, or logic default
        if like is not None:
            dtype = like.dtype
        else:
            dtype = self.valid_A.dtype if self.valid_A.numel() > 0 else torch.float32

        x_full_flat = torch.zeros(self.X * self.Y * self.Z, device=self.device, dtype=dtype)
        x_full_flat[self._free_flat_indices] = x_free.to(dtype)
        return x_full_flat.reshape(self.X, self.Y, self.Z)

    def forward(self, x_free: torch.Tensor) -> torch.Tensor:
        """Compute A(x_full) for x_full expanded from x_free."""
        x_full = self.expand_x(x_free)
        # Call base class forward (which expects (X,Y,Z))
        return super().forward(x_full)

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        """Compute reduced gradient w.r.t free variables.

        Steps:
        - g_full = A^T residual (X,Y,Z)
        - return g_full restricted to free variables.
        """
        g_full = super().adjoint(residual)
        g_free = self.pack_x(g_full)
        return g_free
