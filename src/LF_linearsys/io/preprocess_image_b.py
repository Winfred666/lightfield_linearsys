"""b (2D measurement) denoising utilities.

We keep this file dependency-light (torch-only) so it runs in the same
environment as the rest of the repo.

Pipeline (as requested):
1) morphological opening (erosion then dilation) to remove small bright spots
2) histogram-based background removal

Conventions:
- We expect b images in (H, W) == (Y, X).
- Input/Output are float torch tensors (CPU or GPU). The function is pure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_2d_float(img: torch.Tensor) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"img must be a torch.Tensor, got {type(img)}")
    if img.ndim != 2:
        raise ValueError(f"img must be 2D (H,W), got shape={tuple(img.shape)}")
    return img.float()


def _min_pool2d_replicate(x_hw: torch.Tensor, k: int) -> torch.Tensor:
    # Replicate-pad then min-pool via negative max-pool.
    if k <= 1:
        return x_hw
    pad = k // 2
    x = x_hw.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    y = -F.max_pool2d(-x, kernel_size=k, stride=1)
    return y.squeeze(0).squeeze(0)


def _max_pool2d_replicate(x_hw: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x_hw
    pad = k // 2
    x = x_hw.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    y = F.max_pool2d(x, kernel_size=k, stride=1)
    return y.squeeze(0).squeeze(0)


def _opening_opt(img: torch.Tensor, *, kernel_size: int = 3) -> torch.Tensor:
    """Morphological opening to suppress small bright spots.

    Opening = erosion followed by dilation.
    We implement erosion/dilation using min/max pooling with replicate borders.

    Args:
        img: (H,W) float tensor.
        kernel_size: odd int >= 1.
    """
    img = _as_2d_float(img)
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    eroded = _min_pool2d_replicate(img, kernel_size)
    opened = _max_pool2d_replicate(eroded, kernel_size)
    return opened


def _remove_background(
    img: torch.Tensor,
    *,
    bin_width: float = 0.5,
    max_value_for_hist: float | None = None,
    keep_below_threshold: bool = False,
) -> torch.Tensor:
    """Zero-out (or subtract) low-level background.

    Strategy:
    - Build a histogram (finite pixels only).
    - Find the *first* prominent low-intensity peak (argmax in a small prefix).
    - Set threshold at the first valley after that peak.

    This is intentionally simple and robust for the repo's TIFF measurements.

    Args:
        img: (H,W) float tensor.
        bin_width: histogram bin width in intensity units.
        max_value_for_hist: optional cap to ignore extreme bright outliers.
        keep_below_threshold: if True, keep background and **remove foreground**.

    Returns:
        Background-removed image (same shape/dtype/device).
    """
    img = _as_2d_float(img)
    x = img
    finite = torch.isfinite(x)
    if not torch.any(finite):
        return torch.zeros_like(x)

    values = x[finite]
    vmin = torch.min(values)
    vmax = torch.max(values)
    if max_value_for_hist is not None:
        vmax = torch.minimum(vmax, torch.tensor(float(max_value_for_hist), device=x.device, dtype=x.dtype))
        values = torch.clamp(values, min=float(vmin.item()), max=float(vmax.item()))

    bw = float(bin_width)
    if bw <= 0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")

    # If image is essentially constant, bail.
    if float((vmax - vmin).abs().item()) <= 1e-12:
        thr = float(vmin.item())
        if keep_below_threshold:
            return torch.where(x <= thr, x, torch.zeros_like(x))
        return torch.where(x > thr, x, torch.zeros_like(x))

    nbins = int(torch.ceil((vmax - vmin) / bw).item()) + 1
    nbins = max(nbins, 16)
    nbins = min(nbins, 4096)

    hist = torch.histc(values, bins=nbins, min=float(vmin.item()), max=float(vmax.item()))

    # Search for a background peak in the first 20% bins (at least 10 bins).
    prefix = max(10, int(0.2 * nbins))
    prefix = min(prefix, nbins)
    peak_idx = int(torch.argmax(hist[:prefix]).item())

    # Find first local minimum (valley) after the peak.
    thr_idx = min(peak_idx + 1, nbins - 1)
    for i in range(peak_idx + 1, nbins - 1):
        if hist[i] <= hist[i - 1] and hist[i] <= hist[i + 1]:
            thr_idx = i
            break

    thr = float(vmin.item() + thr_idx * bw)
    if keep_below_threshold:
        return torch.where(x <= thr, x, torch.zeros_like(x))
    return torch.where(x > thr, x, torch.zeros_like(x))


def denoise_image_b(
    img: torch.Tensor,
    *,
    opening_kernel_size: int = 3,
    background_bin_width: float = 0.1,
) -> torch.Tensor:
    """Denoise a measurement image b.

    Applies:
    1) opening
    2) background removal
    """
    img = _as_2d_float(img)
    x = _opening_opt(img, kernel_size=opening_kernel_size)
    x = _remove_background(x, bin_width=background_bin_width)
    return x
