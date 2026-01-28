import torch
import logging
from .linear_system_pair import LinearSystemPair
from ..io.data_postclean import post_clean_row

logger = logging.getLogger(__name__)

class LinearSystemPoint:
    """
    A LinearSystem specialized for a single spatial point (x, y).
    Solves for x in R^Z.
    System is Ax = b, where A is (N_measurements, Z), b is (N_measurements).
    """
    def __init__(self, point_data_path, device='cpu', threshold_A=None, threshold_b=None):
        self.device = device
        self.threshold_A = threshold_A
        self.threshold_b = threshold_b
        self.load_data(point_data_path)
        
    def load_data(self, path):
        logger.info(f"Loading point system from {path}")
        data = torch.load(path, map_location=self.device)
        
        self.A = data['A'].to(self.device).float() # (N, Z)
        self.b = data['b'].to(self.device).float() # (N,)
        self.coord = data.get('coord', (0, 0))
        
        # Apply post_clean_row
        idx, self.A, self.b = post_clean_row(self.A, self.b, self.threshold_A, self.threshold_b)
        
        self.N, self.Z = self.A.shape
        # Conceptually X=1, Y=1 for the solver/plotter compatibility
        self.X = 1
        self.Y = 1
        
        logger.info(f"Point System loaded. Measurements: {self.N} (filtered from {len(idx)}?), Depth(Z): {self.Z}")
        
    def forward(self, x):
        """
        x: (Z,) or (1, 1, Z)
        Returns: (N,)
        """
        # Handle input shapes
        if x.dim() == 3: # (1, 1, Z)
            x_vec = x.reshape(-1)
        else:
            x_vec = x
            
        return torch.matmul(self.A, x_vec)
        
    def adjoint(self, residual):
        """
        residual: (N,)
        Returns: (Z,)
        """
        # A^T * r
        res_vec = torch.matmul(self.A.T, residual)
        return res_vec.reshape(self.Z)
    
    def pack_x(self, x_full):
        """
        Optional: If we wanted to support masking.
        For now, just return flat x.
        """
        return x_full.reshape(-1)

    def expand_x(self, x_free):
        """
        Reshape back to (1, 1, Z)
        """
        return x_free.reshape(1, 1, self.Z)
