import torch
import logging
from typing import Iterable, Union

logger = logging.getLogger(__name__)

def post_clean_row(A: torch.Tensor, b: torch.Tensor, threshold_A: float = None, threshold_b: float = None):
    """
    Filter rows of linear system Ax = b based on thresholds.
    Args:
        A: (N, Z) tensor of system matrix.
        b: (N,) tensor of targets.
        threshold_A: If provided, keep rows where max(abs(row)) > threshold_A.
        threshold_b: If provided, keep rows where b > threshold_b.
    Returns:
        valid_indices: (K,) indices of kept rows in original N.
        A_clean: (K, Z)
        b_clean: (K,)
    """
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"Shape mismatch in post_clean_row: A {A.shape} vs b {b.shape}")

    N = A.shape[0]
    device = A.device
    
    # Start with all valid
    mask = torch.ones(N, dtype=torch.bool, device=device)
    
    # 1. Threshold on A
    if threshold_A is not None:
        # Calculate max absolute value per row
        row_max = torch.max(torch.abs(A), dim=1).values
        mask_A = row_max > threshold_A
        mask = mask & mask_A
        
    # 2. Threshold on b: clip to zero instead of filtering
    if threshold_b is not None:
        b = torch.where(b <= threshold_b, torch.zeros_like(b), b)
        
    valid_indices = torch.nonzero(mask).reshape(-1)
    
    # Logging stats
    kept = valid_indices.numel()
    if kept < N:
        logger.debug(f"post_clean_row: kept {kept}/{N} rows ({(kept/N)*100:.2f}%). "
                     f"Th_A={threshold_A}, Th_b={threshold_b} (clipped b to zero)")
    
    return valid_indices, A[valid_indices], b[valid_indices]


def compute_valid_z_indices(
    A: Union[torch.Tensor, Iterable[torch.Tensor]],
    threshold_A: float = 1e-6,
) -> torch.Tensor:
    """Return the union of valid z indices where |A| has activity above threshold.

    This is a more general version of `compute_active_z_range` suited for joint solving
    multiple pairs (A_list). Instead of returning a contiguous [z_min, z_max) range, it
    returns the exact set of z indices that are active in *any* A.

    Args:
        A: Either a single tensor of shape (..., Z) or an iterable of such tensors.
        threshold_A: activity threshold.

    Returns:
        1D LongTensor of sorted unique z indices (on CPU). Empty tensor if none.
    """
    if isinstance(A, torch.Tensor):
        A_list = [A]
    else:
        A_list = list(A)

    if not A_list:
        return torch.empty((0,), dtype=torch.long)

    # Determine Z from first tensor
    Z = int(A_list[0].shape[-1])
    valid_mask = torch.zeros((Z,), dtype=torch.bool)

    for Ai in A_list:
        if not isinstance(Ai, torch.Tensor) or Ai.numel() == 0:
            continue
        if int(Ai.shape[-1]) != Z:
            raise ValueError(f"compute_valid_z_indices: Z mismatch: {Ai.shape[-1]} vs {Z}")

        # Flatten to (N, Z)
        A_flat = Ai.reshape(-1, Z)
        z_max_vals = A_flat.abs().max(dim=0).values.detach().cpu()
        valid_mask |= (z_max_vals > threshold_A)

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).reshape(-1).long()
    if valid_indices.numel() == 0:
        logger.warning("compute_valid_z_indices: No active Z slices found.")
        return valid_indices

    logger.info(
        "Valid Z indices: %d/%d (min=%d, max=%d)",
        int(valid_indices.numel()),
        int(Z),
        int(valid_indices.min().item()),
        int(valid_indices.max().item()),
    )
    return valid_indices

def compute_active_z_range_from_indices(valid_z_indices: torch.Tensor) -> tuple[int, int]:
    """Helper: convert valid z indices to a [z_min, z_max) range."""
    if valid_z_indices is None or valid_z_indices.numel() == 0:
        return 0, 0
    z_min = int(valid_z_indices.min().item())
    z_max = int(valid_z_indices.max().item()) + 1
    return z_min, z_max
