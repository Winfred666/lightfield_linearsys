import torch
import logging

logger = logging.getLogger(__name__)

class PointLinearSystem:
    """
    Batched Linear System for Point-wise Light Field Reconstruction.
    Solves Ax = b for a batch of points simultaneously.
    
    A: (Batch, N_pairs, Z)
    b: (Batch, N_pairs)
    x: (Batch, Z)
    """
    def __init__(self, A: torch.Tensor, b: torch.Tensor, device: torch.device, threshold_A: float = None, threshold_b: float = None):
        self.device = device
        self.A = A.to(device) # (B, M, N)
        self.b = b.to(device) # (B, M)
        
        # Apply masking based on rows (zero out invalid rows to maintain batch shape)
        # IMPORTANT semantics:
        # - threshold_A controls whether a measurement row is considered valid.
        # - threshold_b should NOT be used to drop rows (b can legitimately be 0).
        #   Instead, threshold_b only suppresses b-values below threshold.
        mask = torch.ones_like(self.b, dtype=torch.bool)

        if threshold_A is not None:
            # A is (B, M, N). Max over N (dim 2)
            row_max = torch.max(torch.abs(self.A), dim=2).values
            mask = row_max > threshold_A

        # Apply row mask to A always (and to b as well, to keep equation consistent)
        if threshold_A is not None:
            self.A = self.A * mask.unsqueeze(2).to(self.A.dtype)
            self.b = self.b * mask.to(self.b.dtype)

        # Apply threshold_b only to b values (do not affect row validity)
        if threshold_b is not None:
            self.b = torch.where(self.b > threshold_b, self.b, torch.zeros_like(self.b))

        if threshold_A is not None:
            total = mask.numel()
            kept = int(mask.sum().item())
            logger.info(
                "PointLinearSystem: Masked %d/%d rows (%.2f%% kept). Th_A=%s, Th_b=%s",
                total - kept,
                total,
                (kept / max(total, 1)) * 100.0,
                threshold_A,
                threshold_b,
            )

        self.B, self.M, self.N = self.A.shape
        
        # Precompute A^T for adjoint
        self.At = self.A.transpose(1, 2) # (B, N, M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Ax.
        x: (B, N)
        Returns: (B, M)
        """
        # (B, M, N) @ (B, N, 1) -> (B, M, 1) -> squeeze -> (B, M)
        return torch.bmm(self.A, x.unsqueeze(2)).squeeze(2)

    def adjoint(self, r: torch.Tensor) -> torch.Tensor:
        """
        Computes A^T r.
        r: (B, M)
        Returns: (B, N)
        """
        # (B, N, M) @ (B, M, 1) -> (B, N, 1) -> squeeze -> (B, N)
        return torch.bmm(self.At, r.unsqueeze(2)).squeeze(2)
        
    def estimate_lipschitz(self, num_iters=10) -> torch.Tensor:
        """
        Estimates Lipschitz constant (max eigenvalue of A^T A) for each batch item 
        using Power Iteration.
        
        Returns: L (Batch,)
        """
        # Initialize random vector
        x = torch.randn(self.B, self.N, device=self.device)
        x = x / torch.norm(x, dim=1, keepdim=True)
        
        for _ in range(num_iters):
            # Apply A^T A
            Ax = self.forward(x)
            AtAx = self.adjoint(Ax)
            
            # Normalize
            norms = torch.norm(AtAx, dim=1, keepdim=True)
            x = AtAx / (norms + 1e-8)
            
        # Rayleight quotient approximation: ||A x||^2 / ||x||^2 approx ||A x||^2 since ||x||=1
        # Actually max eigenvalue of A^T A is approx || A x || / ||x|| is not quite right if we normalized AtAx
        # The singular value is sigma = ||A v||. The eigenvalue of A^T A is sigma^2.
        
        # Let's check:
        # Power iter for M = A^T A.
        # v_{k+1} = M v_k / ||M v_k||.
        # lambda approx v_k^T M v_k = v_k^T A^T A v_k = ||A v_k||^2.
        
        # We have x approx eigenvector of A^T A.
        # Compute || A x ||^2
        Ax = self.forward(x)
        L = torch.sum(Ax**2, dim=1) # (B,)
        
        return L

