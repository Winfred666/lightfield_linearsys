# Light Field Linear Solver

This repository implements solvers (ISTA, FISTA, Newton) for large-scale sparse linear systems ($Ax=b$) arising from light-field microscopy reconstruction.

## Key Features

- **Solvers:** ISTA, FISTA, and Newton (Active Set) methods.
- **Sparse Linear System:** Efficient handling of large sparse matrices.
- **On-the-fly Processing:** Directly reconstructs from raw volume/image pairs, avoiding large intermediate files.
- **Visualization:** Tools for inspecting density volumes, reprojections, and raw data stats.

## Directory Structure

- `src/`: Core solver logic and data utilities.
- `scripts/`: Visualization and maintenance scripts.
- `data/`: Data directory.
    - `data/raw/`: Place your raw `Interp_Vol_ID_*.pt` and `1scan (*).tif` files here.
- `config/`: YAML configuration files for different solvers.
- `result/`: Output directory for reconstruction results.

## Quick Start

### 1. Environment Setup

Ensure you have a conda environment with PyTorch installed.

```bash
conda activate torch
pip install -e .
```

### 2. Data Preparation

Place your raw data in `data/raw/`:
- **Light Field Volumes:** `data/raw/lightsheet_vol_6.9/Interp_Vol_ID_*.pt`
- **Target Images:** `data/raw/20um_imgs/1scan (*).tif`

### 3. Reconstruction (On-the-fly)

We recommend running the solvers directly on raw data. This avoids generating massive intermediate HDF5 files.

**Running ISTA (Pair-based):**

Edit `config/solve_pair_ista_20.yaml` to point to your raw data:
```yaml
data:
  raw_A_dir: "data/raw/lightsheet_vol_6.9"
  raw_b_dir: "data/raw/20um_imgs"
  downsampling_rate: 0.125
  output_dir: "result/solve_pair/ista_onthefly"
```

Run the driver:
```bash
python driver_pair.py --config config/solve_pair_ista_20.yaml
```

**Running Newton (Point-based):**

Edit `config/solve_point_newton_20.yaml`:
```yaml
data:
  raw_A_dir: "data/raw/lightsheet_vol_6.9"
  raw_b_dir: "data/raw/20um_imgs"
  downsampling_rate: 0.5
  output_dir: "result/solve_point/newton_onthefly"
```

Run the driver:
```bash
python driver_point.py --config config/solve_point_newton_20.yaml
```

### 4. Visualization

**Visualize Reconstruction & Reprojection Error:**

This script loads the reconstruction result and re-projects it using the raw data to compare against the target images.

```bash
python scripts/visualize_density_slices.py result/your_experiment/reconstruction.pt \
    --raw-A-dir data/raw/lightsheet_vol_6.9 \
    --raw-b-dir data/raw/20um_imgs \
    --downsampling-rate 0.125 \
    --stride-pairs 5
```

**Visualize Raw Data (Volume & Image stats):**

Directly inspect the raw data pairs without creating HDF5 files.

```bash
python scripts/visualize_raw.py \
    --input-dir data/raw/lightsheet_vol_6.9 \
    --img-dir data/raw/20um_imgs \
    --downsampling-rate 0.125
```

## Legacy Preprocessing (Deprecated)

*Note: The old workflow involved converting raw data into `pair_*.h5` files or `points_batch_*.pt` files. This is now deprecated as it consumes significant disk space and I/O time.*

If you still need to use pre-processed files:
1. Run `src/io/preprocess_pair.py` or `src/io/preprocess_point.py` to generate intermediate files.
2. Update config files to use `data_dir` (for pairs) or `points_dir` (for points) instead of `raw_*_dir`.

## Troubleshooting

- **OOM Errors:** Decrease `file_batch_size` (points) or `joint_pair_num` (pairs) in the config.
- **CUDA Errors:** Ensure `device: "cuda"` is set in config and you have a valid GPU.
