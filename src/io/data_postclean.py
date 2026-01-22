import torch
import logging

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


def compute_active_z_range(A: torch.Tensor, threshold_A: float = 1e-6) -> tuple[int, int]:
    """
    Computes the Z-range [z_min, z_max) where A has significant values.
    
    Args:
        A: Tensor of shape (..., Z). Will be flattened to (N, Z) internally for check.
        threshold: Value threshold.
        
    Returns:
        (z_min, z_max) indices. If no values > threshold, returns (0, 0).
    """
    if A.numel() == 0:
        return 0, 0
        
    Z = A.shape[-1]
    # Flatten to (N, Z)
    A_flat = A.reshape(-1, Z)
    
    # Max over rays (N) -> (Z,)
    # Check if ANY ray has value > threshold at this depth z
    z_max_vals = A_flat.abs().max(dim=0).values
    
    valid_mask = z_max_vals > threshold_A
    valid_indices = torch.nonzero(valid_mask).squeeze()
    
    if valid_indices.numel() == 0:
        logger.warning("compute_active_z_range: No active Z slices found.")
        return 0, 0
        
    z_min = valid_indices.min().item()
    z_max = valid_indices.max().item() + 1 # Exclusive upper bound
    
    logger.info(f"Active Z range: [{z_min}, {z_max}) out of {Z}")
    return z_min, z_max
