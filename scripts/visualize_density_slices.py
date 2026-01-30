#!/usr/bin/env python3

"""Visualize reconstruction slices and reprojection comparisons.

This script is a CLI wrapper around
`LF_linearsys.utils.visualize_slices.visualize_reconstruction_and_reprojection`.

It supports:
- Processed mode: `--data-dir` with `pair_*.h5`
- Raw mode: `--raw-A-dir` and `--raw-b-dir` (preprocess on the fly)

Outputs are written to `<run_dir>/viz` by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from LF_linearsys.utils.visualize_slices import visualize_reconstruction_and_reprojection


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_reconstruction_path(p: Path) -> Path:
    """Accept either a reconstruction.pt file or a directory containing one."""
    if p.is_dir():
        cand = p / "reconstruction.pt"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"No reconstruction.pt found in directory: {p}")
    return p


def _load_reconstruction_volume(pt_path: Path) -> torch.Tensor:
    logger.info(f"Loading reconstruction from {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, dict) and "reconstruction" in data:
        vol = data["reconstruction"]
    else:
        vol = data
    return vol.detach().cpu().float()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize reconstruction slices and reprojection error (shared utility wrapper)."
    )
    parser.add_argument("input", nargs="?", help="Path to reconstruction.pt or directory")

    parser.add_argument(
        "--threshold-A",
        type=float,
        default=0.1,
        help="Threshold used for valid-z and overlay contour (default: 0.1)",
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing pair_*.h5 files (Processed mode)",
    )

    parser.add_argument("--raw-A-dir", default=None, help="Raw volume directory")
    parser.add_argument("--raw-b-dir", default=None, help="Raw image directory")
    parser.add_argument("--downsampling-rate", type=float, default=0.125)
    parser.add_argument("--scale-factor", type=float, default=8.0)

    parser.add_argument("--output-dir", default=None, help="Output directory (default: <run_dir>/viz)")
    parser.add_argument("--stride-pairs", type=int, default=1)
    parser.add_argument("--no-z-scan", action="store_true", help="Disable Z-scan MP4 generation")

    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        recons = sorted(list(Path("result").glob("**/reconstruction.pt")))
        if not recons:
            print("No reconstruction.pt found in result/ and no input provided.")
            sys.exit(1)
        input_path = recons[-1]

    pt_path = _resolve_reconstruction_path(input_path.expanduser().resolve())
    run_dir = pt_path.parent
    viz_dir = Path(args.output_dir) if args.output_dir else (run_dir / "viz")

    vol = _load_reconstruction_volume(pt_path)
    logger.info(f"Reconstruction loaded. Shape: {tuple(vol.shape)}, Max: {vol.max():.4f}")

    visualize_reconstruction_and_reprojection(
        vol=vol,
        output_dir=viz_dir,
        threshold_A=float(args.threshold_A),
        data_dir=args.data_dir,
        raw_A_dir=args.raw_A_dir,
        raw_b_dir=args.raw_b_dir,
        downsampling_rate=float(args.downsampling_rate),
        scale_factor=float(args.scale_factor),
        stride_pairs=int(args.stride_pairs),
        make_z_scan_video=not bool(args.no_z_scan),
    )

    logger.info(f"Visualization complete. Results in {viz_dir}")


if __name__ == "__main__":
    main()
