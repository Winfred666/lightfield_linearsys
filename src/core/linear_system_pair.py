import torch
import logging
from typing import List, Tuple, Optional
from src.io.data_postclean import post_clean_row

logger = logging.getLogger(__name__)

class LinearSystemPair:
    """
    Handles the combined system of multiple Ax=b pairs.
    Optimized to filter out zero rows.
    """
    def __init__(self, A_list: List[torch.Tensor], b_list: List[torch.Tensor], device='cpu', threshold_A=1e-6, threshold_b=None):
        """
        A_list: List of tensors of shape (X, Y, Z).
        b_list: List of tensors of shape (Y, X).
        """
        self.device = device
        self.indices_list = []
        self.values_list = []
        self.targets_list = []
        
        # We assume all As have same spatial dimensions (X, Y) and Z
        # We process them to extract only valid "rows" (pixels)
        
        self.setup(A_list, b_list, threshold_A=threshold_A, threshold_b=threshold_b)
    
    
    def setup(self, A_list, b_list, threshold_A=1e-6, threshold_b=None):
        """
        Converts dense tensors to a sparse-like list of valid pixels.
        """
        if not A_list:
            return
            
        shape_A = A_list[0].shape
        self.shape_x = shape_A

        # WARNING: X, Y, Z is only the range of dot product, not linear system shape which is (B*HW x WHD)
        self.X, self.Y, self.Z = shape_A
        
        all_indices = []
        all_values = []
        all_b = []
        
        total_rows = 0
        kept_rows = 0
        
        # Iterate and consume
        for i in range(len(A_list)):
            A = A_list[i] # shape (X, Y, Z)
            b = b_list[i] # shape (Y, X)

            A_flat = A.reshape(-1, self.Z) # shape (X*Y, Z)
            # IMPORTANT: A is stored as (X, Y, Z) while b is stored as (Y, X).
            # To align rows/pixels between A and b under a shared flat index in [0, X*Y),
            # we must make b have the same (X, Y) memory layout before flattening.
            # In tests and preprocessing, b is consistently treated as (Y, X) (image-style).
            # Therefore, transpose it to (X, Y) first.
            b_flat = b.transpose(0, 1).reshape(-1)  # shape (X*Y,)

            valid_indices, valid_A_rows, valid_b_rows = post_clean_row(A_flat, b_flat, threshold_A=threshold_A, threshold_b=threshold_b)
            
            if valid_indices is None:
                # empty
                n_valid = 0
            else:
                n_valid = valid_indices.numel()
            
            # A_flat shape was calculated inside helper
            n_total = A.shape[0] * A.shape[1] # Approximate total

            if n_valid > 0:
                # Store in float16
                valid_A_rows = valid_A_rows.to(torch.float16)
                valid_b_rows = valid_b_rows.to(torch.float16)

                all_indices.append(valid_indices) # shaped (N_valid_1 + N_valid_2 + ...,), store flat indices range 0,(X*Y)
                all_values.append(valid_A_rows) # shaped (N_valid_1 + N_valid_2 + ..., Z)
                all_b.append(valid_b_rows) # shaped (N_valid_1 + N_valid_2 + ...,)
                kept_rows += n_valid

            total_rows += n_total
            
            # Free memory, only necessary when GPU OOM.
            # A_list[i] = None
            # b_list[i] = None
            # import gc
            # gc.collect()

        logger.info(f"Compressed system: kept {kept_rows}/{total_rows} rows ({(kept_rows/total_rows)*100:.2f}%)")
        
        if kept_rows > 0:
            self.valid_indices = torch.cat(all_indices) if len(all_indices) > 1 else all_indices[0]
            self.valid_A = torch.cat(all_values) if len(all_values) > 1 else all_values[0]
            self.valid_b = torch.cat(all_b) if len(all_b) > 1 else all_b[0]
            
            self.valid_indices = self.valid_indices.to(self.device)
            self.valid_A = self.valid_A.to(self.device)
            self.valid_b = self.valid_b.to(self.device)
            
            self.b = self.valid_b
        else:
            raise ValueError("No valid rows found!")

    def _ensure_device(self, x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Ensure x is on the same device as the system.

        This prevents hard-to-debug RuntimeError: tensors on different devices.
        We log a warning because moving large tensors every call can be expensive.
        """
        if x.device != torch.device(self.device):
            logger.warning(
                "LinearSystem: moving %s from %s to %s to match system device.",
                name,
                x.device,
                self.device,
            )
            return x.to(self.device)
        return x

    def forward(self, x):
        """
        Computes Ax.
        x: (X, Y, Z)
        Returns: vector of predictions corresponding to valid rows.
        """
        # x = self._ensure_device(x, name="x")
        # x shape (X, Y, Z) -> flat (X*Y, Z)
        x_flat = x.reshape(-1, self.Z)
        
        # We need to gather x for the valid indices.
        # valid_indices contains indices in range [0, X*Y).
        # x_sub: (N_valid, Z), just broadcast long as self.valid_A does.
        x_sub = x_flat[self.valid_indices]
        
        # Element-wise dot product between valid_A and x_sub
        # valid_A: (N_valid, Z)
        # x_sub: (N_valid, Z)
        # result: (N_valid)
        result = torch.sum(self.valid_A * x_sub, dim=1)
        return result

    def adjoint(self, residual):
        """
        Computes A^T * residual.
        residual: vector of size (N_valid)
        Returns: gradient w.r.t x, shape (X, Y, Z)
        """
        # residual = self._ensure_device(residual, name="residual")
        # residual: (N_valid)
        # valid_A: (N_valid, Z)
        
        # weighted_A = residual[:, None] * valid_A  -> (N_valid, Z)
        weighted_A = residual.unsqueeze(1) * self.valid_A
        
        # Now we need to scatter_add these back to the full x grid.
        # Output shape: (X*Y, Z)
        grad_flat = torch.zeros(self.X * self.Y, self.Z, device=self.device, dtype=weighted_A.dtype)
        
        # valid_indices: (N_valid)
        # We need to add weighted_A[i] to grad_flat[valid_indices[i]]
        
        # index_expanded: (N_valid, Z)
        index_expanded = self.valid_indices.unsqueeze(1).expand(-1, self.Z)
        
        grad_flat.scatter_add_(0, index_expanded, weighted_A)
        
        return grad_flat.reshape(self.X, self.Y, self.Z)