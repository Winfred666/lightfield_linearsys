#!/usr/bin/env python3
"""Convert EMD (HDF5) reconstruction files to .pt format.

This script reads EMD files from the MLP_result folder and converts them
to .pt files with 'reconstruction' key representing the volume.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_emd_file(emd_path: Path) -> np.ndarray:
    """Read reconstruction volume from EMD (HDF5) file.

    Args:
        emd_path: Path to .emd file

    Returns:
        NumPy array containing reconstruction volume
    """
    logger.info(f"Reading EMD file: {emd_path}")

    try:
        with h5py.File(emd_path, 'r') as f:
            # Try common dataset names for reconstruction data
            # EMD files often store data in datasets like '/data' or '/reconstruction'
            dataset_names = []
            f.visit(lambda name: dataset_names.append(name) if isinstance(f[name], h5py.Dataset) else None)

            logger.info(f"Available datasets in {emd_path.name}: {dataset_names}")

            # Look for likely dataset names
            for dataset_name in ['/data', '/reconstruction', '/volume', '/Image']:
                if dataset_name in f:
                    data = f[dataset_name][:]
                    logger.info(f"Found dataset '{dataset_name}' with shape {data.shape}, dtype {data.dtype}")
                    return data

            # If no standard names found, try to find the first 3D dataset
            for dataset_name in dataset_names:
                dataset = f[dataset_name]
                if isinstance(dataset, h5py.Dataset) and len(dataset.shape) == 3:
                    data = dataset[:]
                    logger.info(f"Using 3D dataset '{dataset_name}' with shape {data.shape}, dtype {data.dtype}")
                    return data

            # If still not found, try the first dataset
            if dataset_names:
                dataset_name = dataset_names[0]
                data = f[dataset_name][:]
                logger.info(f"Using first dataset '{dataset_name}' with shape {data.shape}, dtype {data.dtype}")
                return data

            raise ValueError(f"No suitable dataset found in {emd_path}")

    except Exception as e:
        logger.error(f"Error reading {emd_path}: {e}")
        raise


def convert_emd_to_pt(emd_path: Path, output_dir: Path) -> Path:
    """Convert single EMD file to .pt format.

    Args:
        emd_path: Path to input .emd file
        output_dir: Directory to save .pt file

    Returns:
        Path to created .pt file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename (replace .emd with .pt)
    output_name = emd_path.stem + ".pt"
    output_path = output_dir / output_name

    # Read EMD file
    volume_np = read_emd_file(emd_path)

    # Convert to torch tensor
    volume_tensor = torch.from_numpy(volume_np).float()

    # Save as .pt file with 'reconstruction' key
    torch.save({"reconstruction": volume_tensor}, output_path)
    logger.info(f"Saved reconstruction to {output_path} (shape: {volume_tensor.shape})")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EMD (HDF5) reconstruction files to .pt format"
    )
    parser.add_argument(
        "--input-dir",
        default="/home/ym.xiao/workspace/lightfield_linearsys/result/MLP_result",
        help="Directory containing .emd files (default: result/MLP_result)"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ym.xiao/workspace/lightfield_linearsys/result/MLP_result/pt_files",
        help="Directory to save .pt files (default: result/MLP_result/pt_files)"
    )
    parser.add_argument(
        "--pattern",
        default="*.emd",
        help="Glob pattern for EMD files (default: *.emd)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find all EMD files
    emd_files = list(input_dir.glob(args.pattern))
    if not emd_files:
        logger.error(f"No EMD files found in {input_dir} with pattern '{args.pattern}'")
        sys.exit(1)

    logger.info(f"Found {len(emd_files)} EMD files to convert:")
    for emd_file in emd_files:
        logger.info(f"  - {emd_file.name}")

    # Convert each EMD file
    pt_files = []
    for emd_file in emd_files:
        try:
            pt_path = convert_emd_to_pt(emd_file, output_dir)
            pt_files.append(pt_path)
        except Exception as e:
            logger.error(f"Failed to convert {emd_file.name}: {e}")

    logger.info(f"Conversion complete. Created {len(pt_files)} .pt files in {output_dir}")

    # Print summary
    if pt_files:
        logger.info("Created files:")
        for pt_file in pt_files:
            logger.info(f"  - {pt_file.name}")


if __name__ == "__main__":
    main()