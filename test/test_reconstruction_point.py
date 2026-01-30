import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from LF_linearsys.core.batched_newton_activeset import BatchedRegNewtonASSolver
from LF_linearsys.core.fista import FISTASolver
from LF_linearsys.core.point_system import PointLinearSystem


def _make_multiball_volume(
	X: int,
	Y: int,
	Z: int,
	*,
	n_balls: int = 3,
	radius: float | None = None,
	seed: int = 0,
	dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""Create a small synthetic volume made of several additive balls.

	Returns an (X,Y,Z) tensor with nonnegative values.
	"""
	g = torch.Generator(device="cpu")
	g.manual_seed(int(seed))

	if radius is None:
		radius = min(X, Y, Z) / 6
	radius = float(radius)

	vol = torch.zeros((X, Y, Z), dtype=dtype)

	# Random-ish centers with margins to avoid clipping.
	margin = int(max(2, np.ceil(radius + 1)))
	xs = torch.randint(margin, max(margin + 1, X - margin), (n_balls,), generator=g)
	ys = torch.randint(margin, max(margin + 1, Y - margin), (n_balls,), generator=g)
	zs = torch.randint(margin, max(margin + 1, Z - margin), (n_balls,), generator=g)
	amps = torch.rand((n_balls,), generator=g) * 0.8 + 0.2

	# Precompute a coordinate grid once (small sizes only).
	xx, yy, zz = torch.meshgrid(
		torch.arange(X, dtype=torch.float32),
		torch.arange(Y, dtype=torch.float32),
		torch.arange(Z, dtype=torch.float32),
		indexing="ij",
	)
	for cx, cy, cz, a in zip(xs, ys, zs, amps):
		dist2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 + (zz - float(cz)) ** 2
		mask = dist2 <= radius**2
		vol[mask] += float(a)

	return vol


def _make_point_batch_from_volume(
	vol: torch.Tensor,
	*,
	B: int,
	M: int,
	seed: int = 0,
	noise_std: float = 0.0,
	dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Build a batched point system (A,b) from a synthetic volume.

	We synthesize per-point A as random, and define b = A @ x_true.
	x_true is a depth profile sampled from the volume at random (x,y) coordinates.

	Returns:
	  A: (B, M, Z)
	  b: (B, M)
	  x_true: (B, Z)
	"""
	X, Y, Z = map(int, vol.shape)

	g = torch.Generator(device="cpu")
	g.manual_seed(int(seed))

	# Sample B coordinates.
	xs = torch.randint(0, X, (B,), generator=g)
	ys = torch.randint(0, Y, (B,), generator=g)

	# Depth profiles for each sampled point.
	x_true = torch.stack([vol[int(x), int(y), :].to(dtype) for x, y in zip(xs, ys)], dim=0)

	# Random A (Gaussian).
	A = torch.randn((B, M, Z), generator=g, dtype=dtype)
	b = torch.bmm(A, x_true.unsqueeze(2)).squeeze(2)

	if noise_std > 0:
		b = b + noise_std * torch.randn_like(b, generator=g)

	return A, b, x_true

def test_active_set_newton_solver_on_synth_point_dataset():
	"""Smoke test: Active-set batched Newton on a synthetic batched point dataset.

	Requirements:
	  - create a batched point dataset from a synthetic multiball volume
	  - run BatchedRegNewtonASSolver
	  - save results to result/solve/newton_test
	"""
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	out_dir = Path("result/solve/newton_test")
	out_dir.mkdir(parents=True, exist_ok=True)

	# Keep sizes small so this stays fast in CI.
	X, Y, Z = 16, 24, 32
	B = 8
	M = 24

	vol = _make_multiball_volume(X, Y, Z, n_balls=3, seed=123, dtype=torch.float32)
	A, b, x_true = _make_point_batch_from_volume(vol, B=B, M=M, seed=123, noise_std=0.0, dtype=torch.float32)

	system = PointLinearSystem(A, b, device="cpu")
	solver = BatchedRegNewtonASSolver(
		system,
		lambda_reg=0.01,
		n_iter=30,
		positivity=True,
		output_dir=out_dir,
	)

	x_hat = solver.solve(tag="synth_point_newton")
	assert x_hat.shape == (B, Z)
	assert torch.all(x_hat >= 0)
	assert not torch.isnan(x_hat).any()

	Ax_hat = system.forward(x_hat)
	residual_norm = torch.norm(Ax_hat - b, dim=1)
	logger.info("AS-Newton synth point residual norms: %s", residual_norm)

	# This is a smoke test: we just want it to fit reasonably.
	assert torch.all(residual_norm < 5.0)

	# Save artifacts for manual inspection.
	save_path = out_dir / "synth_point_newton_as.pt"
	torch.save(
		{
			"A": A,
			"b": b,
			"x_true": x_true,
			"x_hat": x_hat,
			"residual_norm": residual_norm,
			"meta": {
				"X": X,
				"Y": Y,
				"Z": Z,
				"B": B,
				"M": M,
				"lambda_reg": 0.01,
				"n_iter": 30,
				"positivity": True,
			},
		},
		save_path,
	)
	assert save_path.exists()

	# Solver should also export a loss curve.
	assert (out_dir / "loss_curve" / "loss_curve_synth_point_newton.png").exists()
