from __future__ import annotations

from pathlib import Path

import torch

from LF_linearsys.io.preprocess_image_b import denoise_image_b
from LF_linearsys.io.readers import read_image
from LF_linearsys.utils.visualize_slices import visualize_reprojection


def _load_tif_as_float_tensor(path: Path) -> torch.Tensor:
	img_np = read_image(str(path))  # (H,W) == (Y,X)
	img = torch.from_numpy(img_np).float()
	return img


def test_denoise_b_writes_comparison_plots() -> None:
	# Inputs (real data files).
	p1 = Path("data/raw/80um_imgs/1scan (1).tif")
	p2 = Path("data/raw/80um_imgs/1scan (9).tif")

	# If the dataset isn't present in CI, skip gracefully.
	if not p1.exists() or not p2.exists():
		import pytest

		pytest.skip("Required raw TIFF files not found under data/raw/80um_imgs")

	out_dir = Path("result/solve/denoise_b")
	out_dir.mkdir(parents=True, exist_ok=True)

	for p in [p1, p2]:
		b = _load_tif_as_float_tensor(p)
		b_dn = denoise_image_b(b)

		# Use visualize_reprojection as a generic compare function:
		# treat original as 'GT b' and denoised as 'b_pred'.
		diff = b - b_dn
		mse = torch.mean(diff ** 2).item()
		data_max = float(b.max().item())
		if data_max <= 0:
			data_max = 1.0
		if mse <= 1e-12:
			psnr = float("inf")
		else:
			psnr = 20.0 * float(torch.log10(torch.tensor(data_max / (mse ** 0.5))).item())

		pair_name = p.stem
		out_path = out_dir / f"compare_{pair_name}.png"
		visualize_reprojection(b, b_dn, mse, psnr, out_path, pair_name=pair_name)

		assert out_path.exists(), f"Expected plot output at {out_path}"
