import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

def read_volume(path, device='cpu'):
    """
    Reads a .pt volume file.
    Args:
        path (str): Path to the .pt file.
        device (str): Device to load the tensor on.
    Returns:
        torch.Tensor: Volume data (X, Y, Z).
    """
    return torch.load(path, map_location=device)

def read_image(path):
    """
    Reads a TIFF image.
    Args:
        path (str): Path to the .tif file.
    Returns:
        numpy.ndarray: Image data (Y, X). 
        Note: PIL opens as (W, H) which is (X, Y). np.array(img) is (H, W) -> (Y, X).
    """
    img = Image.open(path)
    return np.array(img)

def scale_volume(vol, scale_factor):
    """
    Scales the volume using trilinear interpolation.
    
    Args:
        vol (torch.Tensor): Input volume of shape (X, Y, Z).
        scale_factor (float): Scale factor.
        
    Returns:
        torch.Tensor: Scaled volume of shape (X_new, Y_new, Z_new).
    """
    # torch.nn.functional.interpolate expects (N, C, D, H, W).
    # We map our (X, Y, Z) to this.
    # Let's map Z->D, Y->H, X->W.
    # Input vol: (X, Y, Z)
    # Permute to (1, 1, Z, Y, X)
    
    if not isinstance(vol, torch.Tensor):
        vol = torch.tensor(vol)
        
    if vol.dtype != torch.float32:
        vol = vol.float()
        
    d, h, w = vol.shape[2], vol.shape[1], vol.shape[0] # Z, Y, X
    
    # Unsqueeze to (1, 1, Z, Y, X)
    # Note: vol is (X, Y, Z). permute(2, 1, 0) -> (Z, Y, X)
    vol_reshaped = vol.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    
    # Interpolate
    vol_scaled = F.interpolate(
        vol_reshaped, 
        scale_factor=scale_factor, 
        mode='trilinear', 
        align_corners=False
    )
    
    # Squeeze and permute back to (X, Y, Z)
    # (1, 1, Z', Y', X') -> (Z', Y', X') -> (X', Y', Z')
    vol_scaled = vol_scaled.squeeze(0).squeeze(0).permute(2, 1, 0)
    
    return vol_scaled
