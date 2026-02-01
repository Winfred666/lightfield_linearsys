"""LF_linearsys visualization utilities.

This module is a refactor-friendly home for visualization code that was
previously embedded in ad-hoc scripts.

Contract (main API)
-------------------
`visualize_reconstruction_and_reprojection(...)` is a *pure-ish* function:

- Inputs: a loaded reconstruction volume tensor `vol` shaped (X,Y,Z) on CPU,
  plus either:
	* processed pairs via `data_dir` containing `pair_*.h5`, or
	* raw dirs via `raw_A_dir` + `raw_b_dir` with preprocess utilities.
- Outputs: files written under `output_dir` (plots + optional mp4).

It does not parse CLI args and can be called from:
- `driver_pair.py` / `driver_point.py` after saving reconstruction
- the legacy script `visualize_density_slices.py`
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _ensure_cpu_float(vol: torch.Tensor) -> torch.Tensor:
	"""Return a detached CPU float32 tensor."""
	if not isinstance(vol, torch.Tensor):
		raise TypeError(f"vol must be a torch.Tensor, got {type(vol)}")
	return vol.detach().cpu().float()


def numerical_sort_key(p: Path):
	numbers = re.findall(r"\d+", p.name)
	if numbers:
		return int(numbers[-1])
	return p.name


def _robust_vmax(vol: np.ndarray, p: float = 99.0) -> float:
	finite = vol[np.isfinite(vol)]
	if finite.size == 0:
		return 1e-12
	v = float(np.percentile(finite, p))
	return max(v, 1e-12)


def project_and_compare(vol: torch.Tensor, A: torch.Tensor, b: torch.Tensor):
	"""Reproject (A * vol) summed over Z and compare with b.

	Shapes:
	  - vol: (X,Y,Z)
	  - A:   (X,Y,Z)
	  - b:   (Y,X)
	"""
	# Ensure shapes match (handle potential mismatches gracefully)
	if A.shape != vol.shape:
		logger.warning(
			f"Shape mismatch: A {tuple(A.shape)} vs vol {tuple(vol.shape)}. "
			"Truncating to common size."
		)
		sx = min(A.shape[0], vol.shape[0])
		sy = min(A.shape[1], vol.shape[1])
		sz = min(A.shape[2], vol.shape[2])
		A = A[:sx, :sy, :sz]
		vol_use = vol[:sx, :sy, :sz]
	else:
		vol_use = vol

	b_pred_xy = torch.sum(A * vol_use, dim=2)  # (X,Y)
	b_pred = b_pred_xy.T  # (Y,X)

	diff = b - b_pred
	mse = torch.mean(diff**2).item()

	data_max = b.max().item()
	if data_max <= 0:
		data_max = 1.0

	if mse <= 1e-12:
		psnr = float("inf")
	else:
		psnr = 20 * np.log10(data_max / np.sqrt(mse))

	return b_pred, mse, psnr


def visualize_reprojection(b, b_pred, mse, psnr, out_path: Path, pair_name: str):
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

	b_np = b.detach().cpu().numpy()
	pred_np = b_pred.detach().cpu().numpy()

	vmin = min(np.percentile(b_np, 1), np.percentile(pred_np, 1))
	vmax = max(np.percentile(b_np, 99), np.percentile(pred_np, 99))
	if vmax <= vmin:
		vmin, vmax = 0, 1

	im0 = axes[0].imshow(b_np, cmap="viridis", vmin=vmin, vmax=vmax)
	axes[0].set_title(f"GT {pair_name} (b)")
	plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

	im1 = axes[1].imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
	axes[1].set_title(f"Reprojection (A@x)\nMSE={mse:.2e}, PSNR={psnr:.2f} dB")
	plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

	err = pred_np - b_np
	if err.size > 0:
		e = err[np.isfinite(err)]
		if e.size > 0:
			e_max = float(np.percentile(np.abs(e), 99))
			e_max = max(e_max, 1e-12)
		else:
			e_max = 1.0
	else:
		e_max = 1.0

	im2 = axes[2].imshow(err, cmap="coolwarm", vmin=-e_max, vmax=e_max)
	axes[2].set_title("Error (pred - GT)")
	plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

	for ax in axes:
		ax.axis("off")

	plt.suptitle(f"Reprojection Comparison: {pair_name}")
	plt.tight_layout(rect=[0, 0.0, 1, 0.95])
	plt.savefig(out_path)
	plt.close(fig)


def visualize_slices(vol: torch.Tensor, out_path: Path, *, num_slices: int = 25):
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_path.parent.mkdir(parents=True, exist_ok=True)

	vol = _ensure_cpu_float(vol)
	nz = int(vol.shape[2])
	if nz <= 0:
		raise ValueError("Volume has zero Z dimension")

	indices = np.linspace(0, nz - 1, num_slices, dtype=int)

	ncols = int(np.ceil(np.sqrt(num_slices)))
	nrows = int(np.ceil(num_slices / ncols))
	fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
	axes = np.asarray(axes).ravel()

	vol_flat = vol.numpy().ravel()
	if vol_flat.size > 0:
		p99 = np.percentile(vol_flat, 99)
		vmax = max(float(p99), 1e-6)
	else:
		vmax = 1.0

	im = None
	for i, idx in enumerate(indices):
		slice_data = vol[:, :, int(idx)].numpy().T
		im = axes[i].imshow(slice_data, cmap="viridis", vmin=0, vmax=vmax)
		axes[i].set_title(f"Z={int(idx)}")
		axes[i].axis("off")

	for j in range(len(indices), len(axes)):
		axes[j].axis("off")

	if im is not None:
		fig.subplots_adjust(right=0.9)
		cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
		fig.colorbar(im, cax=cbar_ax, label="Density")

	plt.savefig(out_path, bbox_inches="tight")
	plt.close(fig)


def visualize_volume_rendering(vol: torch.Tensor, out_path: Path):
	"""Side view: sum projection along X -> (Y,Z)."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_path.parent.mkdir(parents=True, exist_ok=True)
	vol = _ensure_cpu_float(vol)
	vol_np = vol.numpy()
	proj = np.sum(vol_np, axis=0)  # (Y,Z)

	h, w = proj.shape
	w_in = 6.0
	h_in = max(3.0, w_in * (h / max(w, 1)))
	fig, ax = plt.subplots(figsize=(w_in, h_in))

	if proj.size > 0:
		vmin = float(np.percentile(proj, 1))
		vmax = float(np.percentile(proj, 99))
		if vmax <= vmin:
			vmin, vmax = float(np.min(proj)), float(np.max(proj))
	else:
		vmin, vmax = 0.0, 1.0

	im = ax.imshow(
		proj,
		cmap="viridis",
		aspect="equal",
		origin="lower",
		vmin=vmin,
		vmax=vmax,
	)
	ax.set_title("Side View (Sum projection along X)\nLeft->Right is Z-axis")
	ax.set_xlabel("Z (Depth)")
	ax.set_ylabel("Y (Vertical)")
	fig.colorbar(im, ax=ax, label="Projected density (sum over X)")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close(fig)


def _compute_side_projection_sum_x(vol_xyz: np.ndarray) -> np.ndarray:
	return np.sum(vol_xyz, axis=0)


def _compute_lightfield_energy_stats(A: torch.Tensor, *, threshold_A: float) -> tuple[float, float, float]:
	"""Compute lightfield 'energy' (sum of voxel intensities).

	Returns:
		(total_energy, thresholded_energy, kept_ratio)
	where kept_ratio = thresholded_energy / total_energy (0 if total_energy==0).

	Notes:
	- Uses only finite values.
	- Threshold uses `A >= threshold_A`.
	"""
	A_cpu = A.detach().cpu().float()
	A_np = A_cpu.numpy()
	f = np.isfinite(A_np)
	if not np.any(f):
		return 0.0, 0.0, 0.0

	total = float(A_np[f].sum())
	thr_mask = f & (A_np >= float(threshold_A))
	thr_sum = float(A_np[thr_mask].sum())
	ratio = 0.0 if abs(total) <= 1e-20 else float(thr_sum / total)
	return total, thr_sum, ratio


def visualize_lightfield_intensity(
	stats: list[dict],
	out_path: Path,
	*,
	title: str | None = None,
	annotate_every: int = 1,
) -> None:
	"""Plot per-pair lightfield energy (total vs thresholded) with kept % labels."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_path.parent.mkdir(parents=True, exist_ok=True)
	if not stats:
		logger.warning("No lightfield energy stats provided; skipping intensity plot.")
		return

	# Expect each dict to have: idx, label, energy_total, energy_threshold
	idx = np.asarray([int(d.get("idx", i)) for i, d in enumerate(stats)], dtype=int)
	total = np.asarray([float(d.get("energy_total", 0.0)) for d in stats], dtype=float)
	thr = np.asarray([float(d.get("energy_threshold", 0.0)) for d in stats], dtype=float)

	kept = np.zeros_like(total)
	nz = np.abs(total) > 1e-20
	kept[nz] = thr[nz] / total[nz]

	fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=160)
	ax.plot(idx, total, label="total sum(A)", linewidth=1.6)
	ax.plot(idx, thr, label="sum(A | A>=threshold_A)", linewidth=1.6)
	ax.set_xlabel("pair index")
	ax.set_ylabel("summed intensity")
	if title:
		ax.set_title(title)
	else:
		ax.set_title("Lightfield intensity per pair")
	ax.grid(True, alpha=0.25)
	ax.legend(loc="best")

	# Annotate remaining percentage as small text.
	if annotate_every < 1:
		annotate_every = 1
	for i in range(len(idx)):
		if (i % annotate_every) != 0:
			continue
		pct = 100.0 * float(kept[i])
		y = thr[i]
		ax.text(
			idx[i],
			y,
			f"{pct:.1f}%",
			fontsize=7,
			ha="center",
			va="bottom",
			rotation=0,
			clip_on=True,
			alpha=0.85,
		)

	plt.tight_layout()
	plt.savefig(out_path)
	plt.close(fig)


def visualize_lightfield_side_overlay(
	projections: list[dict],
	out_path: Path,
	*,
	fig_size: tuple[float, float],
	threshold_A: float = 0.1,
	base_alpha: float = 0.01,
	label_fontsize: int = 6,
):
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from matplotlib.colors import to_rgba

	out_path.parent.mkdir(parents=True, exist_ok=True)
	if not projections:
		logger.warning("No lightfield projections provided; skipping overlay plot.")
		return

	cmap = cm.get_cmap("tab20", max(len(projections), 1))
	fig, ax = plt.subplots(figsize=fig_size)

	def _z_key(d: dict):
		zmin = d.get("z_min")
		return (1e18 if zmin is None else int(zmin))

	projections_sorted = sorted(projections, key=_z_key)

	mask_any: np.ndarray | None = None
	for i, item in enumerate(projections_sorted):
		proj = item["proj"]
		if proj.size == 0:
			continue
		finite = proj[np.isfinite(proj)]
		if finite.size == 0:
			continue

		if mask_any is None:
			mask_any = np.zeros_like(proj, dtype=bool)
		mask_any |= np.isfinite(proj) & (proj >= float(threshold_A))

		p = proj.astype(np.float32)
		p_min = float(np.percentile(finite, 1))
		p_max = float(np.percentile(finite, 99))
		if p_max <= p_min:
			p_min = float(np.min(finite))
			p_max = float(np.max(finite))
		denom = max(p_max - p_min, 1e-12)
		p_norm = np.clip((p - p_min) / denom, 0.0, 1.0)

		rgb = np.asarray(cmap(i))[:3]
		alpha = (p_norm * base_alpha).astype(np.float32)

		rgba = np.empty((p.shape[0], p.shape[1], 4), dtype=np.float32)
		rgba[..., 0] = rgb[0]
		rgba[..., 1] = rgb[1]
		rgba[..., 2] = rgb[2]
		rgba[..., 3] = alpha

		ax.imshow(
			rgba,
			origin="lower",
			aspect="equal",
			interpolation="nearest",
		)

	ax.set_title("Side View of Lightfields (Sum projection along X)\nLeft->Right is Z-axis")
	ax.set_xlabel("Z (Depth)")
	ax.set_ylabel("Y (Vertical)")

	if mask_any is not None and np.any(mask_any):
		try:
			ax.contour(
				mask_any.astype(np.float32),
				levels=[0.5],
				colors=["red"],
				linewidths=0.6,
				origin="lower",
			)
		except Exception as e:
			logger.warning(f"Failed to draw threshold mask contour: {e}")

	y_top = projections_sorted[0]["proj"].shape[0] - 1
	for i, item in enumerate(projections_sorted):
		z_min = item.get("z_min")
		z_max = item.get("z_max")
		if z_min is None or z_max is None:
			continue
		z_mid = 0.5 * (float(z_min) + float(z_max))

		y_step = max(1.0, 0.5 * float(label_fontsize))
		y = y_top - i * y_step
		if y < 0:
			break
		color = to_rgba(cmap(i), 1.0)
		ax.text(
			z_mid,
			y,
			item.get("label", f"LF_{i}") + f" [{int(z_min)},{int(z_max)}]",
			fontsize=label_fontsize,
			color=color,
			ha="center",
			va="top",
			bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, pad=1.0),
			clip_on=True,
		)

	plt.tight_layout()
	plt.savefig(out_path)
	plt.close(fig)


def _render_frame_xy(
	density_xy: np.ndarray,
	*,
	density_vmin: float,
	density_vmax: float,
	title: str,
	fig,
	ax,
	canvas,
	add_colorbar: bool,
	cbar_ax,
):
	import matplotlib

	matplotlib.use("Agg")
	import numpy as _np

	ax.clear()
	im = ax.imshow(
		density_xy,
		cmap="viridis",
		vmin=density_vmin,
		vmax=density_vmax,
		origin="lower",
		interpolation="nearest",
	)
	ax.set_title(title)
	ax.set_xlabel("Y")
	ax.set_ylabel("X")

	if add_colorbar and cbar_ax is not None:
		cbar_ax.clear()
		cb = fig.colorbar(im, cax=cbar_ax)
		cb.set_label("Density")

	canvas.draw()

	try:
		w, h = fig.canvas.get_width_height()
		rgb = _np.frombuffer(canvas.tostring_rgb(), dtype=_np.uint8)
		rgb = rgb.reshape((h, w, 3))
		return rgb
	except AttributeError:
		buf = _np.asarray(canvas.buffer_rgba())
		return buf[:, :, :3].copy()


def _write_video_with_temp_frames(
	frames: list[np.ndarray],
	out_path: Path,
	fps: int = 20,
	*,
	delete_frames_after: bool = True,
):
	out_path.parent.mkdir(parents=True, exist_ok=True)

	try:
		import imageio.v2 as imageio
		import tempfile
		import shutil
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		try:
			with imageio.get_writer(str(out_path), fps=fps, format="FFMPEG") as w:
				for fr in frames:
					w.append_data(fr)
			return
		except Exception as e_direct:
			tmp_dir = Path(tempfile.mkdtemp(prefix="zscan_frames_", dir=str(out_path.parent)))
			try:
				for i, fr in enumerate(frames):
					png_path = tmp_dir / f"frame_{i:04d}.png"
					plt.imsave(png_path, fr)

				with imageio.get_writer(str(out_path), fps=fps, format="FFMPEG") as w:
					for i in range(len(frames)):
						fr = imageio.imread(str(tmp_dir / f"frame_{i:04d}.png"))
						w.append_data(fr)

			except Exception as e_seq:
				seq_dir = out_path.with_suffix("")
				if seq_dir.exists():
					shutil.rmtree(seq_dir)
				shutil.move(str(tmp_dir), str(seq_dir))
				logger.warning(
					f"Failed to write video {out_path} (direct={e_direct}; seq={e_seq}). "
					f"Saved PNG sequence to {seq_dir}."
				)
				return
			finally:
				if delete_frames_after and tmp_dir.exists():
					shutil.rmtree(tmp_dir, ignore_errors=True)
	except Exception as e:
		# imageio not installed or no ffmpeg backend
		seq_dir = out_path.with_suffix("")
		seq_dir.mkdir(parents=True, exist_ok=True)
		try:
			import matplotlib

			matplotlib.use("Agg")
			import matplotlib.pyplot as plt

			for i, fr in enumerate(frames):
				plt.imsave(seq_dir / f"frame_{i:04d}.png", fr)
		except Exception:
			pass
		logger.warning(f"Failed to write video {out_path} ({e}). Saved PNGs to {seq_dir}.")


def visualize_z_scan_video(vol: torch.Tensor, out_path: Path, *, fps: int = 20):
	import matplotlib

	matplotlib.use("Agg")
	from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
	import matplotlib.pyplot as plt

	out_path.parent.mkdir(parents=True, exist_ok=True)

	vol = _ensure_cpu_float(vol)
	vol_np = vol.numpy()
	_, _, Z = vol_np.shape

	density_vmin = 0.0
	density_vmax = _robust_vmax(vol_np, p=99.0)

	fig = plt.Figure(figsize=(7.5, 6.5), dpi=140)
	canvas = FigureCanvas(fig)
	gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
	ax = fig.add_subplot(gs[0, 0])
	cax = fig.add_subplot(gs[0, 1])

	frames: list[np.ndarray] = []
	logger.info(f"Rendering {Z} frames for Z-scan video...")
	for z in range(int(Z)):
		density_xy = vol_np[:, :, z]
		title = f"XY slice z={z}/{Z-1}"
		frame = _render_frame_xy(
			density_xy,
			density_vmin=density_vmin,
			density_vmax=density_vmax,
			title=title,
			fig=fig,
			ax=ax,
			canvas=canvas,
			add_colorbar=(z == 0),
			cbar_ax=cax,
		)
		frames.append(frame)

	_write_video_with_temp_frames(frames, out_path, fps=fps, delete_frames_after=True)
	plt.close(fig)
	logger.info(f"Saved Z-scan video to {out_path}")


@dataclass(frozen=True)
class VisualizationOutputs:
	viz_dir: Path
	reproj_dir: Path
	slices_png: Path
	side_png: Path
	zscan_mp4: Path
	lightfield_side_png: Path


def visualize_reconstruction_and_reprojection(
	*,
	vol: torch.Tensor,
	output_dir: str | Path,
	threshold_A: float = 0.1,
	data_dir: str | Path | None = None,
	raw_A_dir: str | Path | None = None,
	raw_b_dir: str | Path | None = None,
	downsampling_rate: float = 0.125,
	scale_factor: float = 8.0,
	stride_pairs: int = 1,
	make_z_scan_video: bool = True,
	z_scan_fps: int = 20,
	lightfield_overlay_alpha: float = 0.06,
	num_slice_grid: int = 25,
	crop_box_A: Iterable[int] | None = None,
	crop_box_b: Iterable[int] | None = None,
) -> VisualizationOutputs:
	"""Main reusable visualization entrypoint.

	Exactly one data source must be provided:
	  - processed: `data_dir` with `pair_*.h5`
	  - raw: `raw_A_dir` and `raw_b_dir`
	"""
	vol = _ensure_cpu_float(vol)
	output_dir = Path(output_dir)
	viz_dir = output_dir
	viz_dir.mkdir(parents=True, exist_ok=True)
	reproj_dir = viz_dir / "reprojection"
	reproj_dir.mkdir(parents=True, exist_ok=True)

	slices_png = viz_dir / "reconstruction_slices.png"
	side_png = viz_dir / "volume_render_side.png"
	zscan_mp4 = viz_dir / "reconstruction_z_scan.mp4"
	lightfield_side_png = viz_dir / "lightfield_render_side.png"

	logger.info("Generating volume slice grid...")
	visualize_slices(vol, slices_png, num_slices=num_slice_grid)

	logger.info("Generating side view rendering...")
	visualize_volume_rendering(vol, side_png)

	if make_z_scan_video:
		logger.info("Generating Z-scan video...")
		visualize_z_scan_video(vol, zscan_mp4, fps=z_scan_fps)

	# Collect lightfield projections for overlay, also collect how much light energy is cutted.
	lightfield_projections: list[dict] = []
	lightfield_energy_stats: list[dict] = []


	raw_mode = raw_A_dir is not None and raw_b_dir is not None
	processed_mode = data_dir is not None
	if raw_mode == processed_mode:
		raise ValueError("Provide either (raw_A_dir, raw_b_dir) OR data_dir.")

	if raw_mode:
		from LF_linearsys.io.data_postclean import compute_valid_z_indices
		from LF_linearsys.io.preprocess_pair import preprocess_one_pair
		from LF_linearsys.io.raw_pairs import find_raw_pairs

		pairs = find_raw_pairs(str(raw_A_dir), str(raw_b_dir))
		if not pairs:
			logger.warning(f"No matching raw pairs found in {raw_A_dir}")
		else:
			pairs = pairs[:: max(int(stride_pairs), 1)]
			logger.info(f"Processing {len(pairs)} raw pairs (stride={stride_pairs})...")
			for p in pairs:
				logger.info(f"Processing Raw Pair Index {p.idx}...")
				A, b = preprocess_one_pair(
					vol_path=p.vol_path,
					img_path=p.img_path,
					downsampling_rate=downsampling_rate,
					scale_factor=scale_factor,
					device=torch.device("cpu"),
					crop_box_A=crop_box_A,
					crop_box_b=crop_box_b,
				)

				valid_z = compute_valid_z_indices(A, threshold_A=threshold_A)
				if valid_z.numel() > 0:
					z_min = int(valid_z.min().item())
					z_max = int(valid_z.max().item())
					pair_label = f"Raw_Pair_{p.idx} [{z_min},{z_max}]"
				else:
					pair_label = f"Raw_Pair_{p.idx} [empty]"
					z_min = None
					z_max = None

				try:
					proj = _compute_side_projection_sum_x(A.detach().cpu().numpy())
					energy_total, energy_thr, kept_ratio = _compute_lightfield_energy_stats(
						A,
						threshold_A=threshold_A,
					)

					lightfield_projections.append(
						{
							"proj": proj,
							"label": f"Raw_{p.idx}",
							"z_min": z_min,
							"z_max": z_max,
							"idx": int(p.idx),
							"energy_total": energy_total,
							"energy_threshold": energy_thr,
							"kept_ratio": kept_ratio,
						}
					)
					lightfield_energy_stats.append(
						{
							"idx": int(p.idx),
							"label": f"Raw_{p.idx}",
							"energy_total": energy_total,
							"energy_threshold": energy_thr,
							"kept_ratio": kept_ratio,
						}
					)
				except Exception as e:
					logger.warning(f"Failed to compute lightfield projection for Raw Pair {p.idx}: {e}")

				b_pred, mse, psnr = project_and_compare(vol, A, b)
				visualize_reprojection(
					b,
					b_pred,
					mse,
					psnr,
					reproj_dir / f"target_image_compare_{p.idx}.png",
					pair_label,
				)

	if processed_mode:
		import h5py
		from LF_linearsys.io.data_postclean import compute_valid_z_indices

		data_dir_p = Path(data_dir)
		if not data_dir_p.exists():
			raise FileNotFoundError(f"Data directory not found: {data_dir_p}")

		pair_files = sorted(list(data_dir_p.glob("pair_*.h5")), key=numerical_sort_key)
		if not pair_files:
			logger.warning(f"No pair_*.h5 files found in {data_dir_p}")
		else:
			pair_files = pair_files[:: max(int(stride_pairs), 1)]
			logger.info(f"Processing {len(pair_files)} pair files (stride={stride_pairs})...")
			for p_file in pair_files:
				stem = p_file.stem
				parts = stem.split("_")
				num = parts[-1] if len(parts) > 1 else "0"

				with h5py.File(p_file, "r") as f:
					A = torch.from_numpy(f["A"][:]).float()
					b = torch.from_numpy(f["b"][:]).float()

				valid_z = compute_valid_z_indices(A, threshold_A=threshold_A)
				if valid_z.numel() > 0:
					z_min = int(valid_z.min().item())
					z_max = int(valid_z.max().item())
					pair_label = f"{stem} [{z_min},{z_max}]"
				else:
					pair_label = f"{stem} [empty]"
					z_min = None
					z_max = None

				try:
					proj = _compute_side_projection_sum_x(A.detach().cpu().numpy())
					energy_total, energy_thr, kept_ratio = _compute_lightfield_energy_stats(
						A,
						threshold_A=threshold_A,
					)
					# numeric id if possible
					try:
						idx_val = int(num)
					except Exception:
						idx_val = int(len(lightfield_energy_stats))

					lightfield_projections.append(
						{
							"proj": proj,
							"label": stem,
							"z_min": z_min,
							"z_max": z_max,
							"idx": idx_val,
							"energy_total": energy_total,
							"energy_threshold": energy_thr,
							"kept_ratio": kept_ratio,
						}
					)
					lightfield_energy_stats.append(
						{
							"idx": idx_val,
							"label": stem,
							"energy_total": energy_total,
							"energy_threshold": energy_thr,
							"kept_ratio": kept_ratio,
						}
					)
				except Exception as e:
					logger.warning(f"Failed to compute lightfield projection for {stem}: {e}")

				b_pred, mse, psnr = project_and_compare(vol, A, b)
				visualize_reprojection(
					b,
					b_pred,
					mse,
					psnr,
					reproj_dir / f"target_image_compare_{num}.png",
					pair_label,
				)

	# Lightfield overlay
	try:
		vol_np = vol.numpy()
		vol_side = _compute_side_projection_sum_x(vol_np)
		h, w = vol_side.shape
		w_in = 6.0
		h_in = max(3.0, w_in * (h / max(w, 1)))
		fig_size = (w_in, h_in)
		logger.info("Generating lightfield side overlay render...")
		visualize_lightfield_side_overlay(
			lightfield_projections,
			lightfield_side_png,
			fig_size=fig_size,
			threshold_A=threshold_A,
			base_alpha=lightfield_overlay_alpha,
		)
	except Exception as e:
		logger.warning(f"Failed to generate lightfield side overlay render: {e}")

	# Lightfield intensity plot
	try:
		if lightfield_energy_stats:
			# Ensure stable ordering by idx when available.
			stats_sorted = sorted(lightfield_energy_stats, key=lambda d: int(d.get("idx", 0)))
			visualize_lightfield_intensity(
				stats_sorted,
				viz_dir / "lightfield_intensity.png",
				title=f"Lightfield intensity per pair (threshold_A={threshold_A})",
				annotate_every=1,
			)
		else:
			logger.warning("No lightfield energy stats collected; skipping intensity plot.")
	except Exception as e:
		logger.warning(f"Failed to generate lightfield intensity plot: {e}")

	return VisualizationOutputs(
		viz_dir=viz_dir,
		reproj_dir=reproj_dir,
		slices_png=slices_png,
		side_png=side_png,
		zscan_mp4=zscan_mp4,
		lightfield_side_png=lightfield_side_png,
	)

