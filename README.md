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

**Running ISTA (Pair-wise):**

Pair-wise solvers use first-order methods, which have has high regularization and cause very smooth/round results.

Edit `config/pair_ista_crop_20um_0p5.yaml` to your raw data, where `crop` means using `crop_box_b` to crop `.tif` measurement, and `crop_box_A` to crop light field so that A_xy * scaling_rate = measurement_xy.  `0p5` means downsampling rate 0.5:

Run the driver:
```bash
python driver_pair.py --config config/pair_ista_crop_20um_0p5.yaml
```

**Running Newton (Point-wise):**

Point-wise solver use second-order methods, which are much faster, more accurate, and sharper, with dynamic regularization.

Edit `config/crop/point_newton_crop_20um_1p0.yaml`:

Run the driver:
```bash
python driver_point.py --config config/crop/point_newton_crop_20um_1p0.yaml
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

## Two-step Preprocessing

*Note: The old workflow involved converting raw data into `pair_*.h5` files or `points_batch_*.pt` files. This is slow as it consumes significant disk space and I/O time.*. 

If you still need to use pre-processed files:

1. Run `src/io/preprocess_pair.py` to generaet `pair_<num>.h5` and `src/io/preprocess_point.py` to generate `points_batch_<num:04d>.pt` as intermediate files.

Crop is supported to both measurement b and A. Make sure that crop ratio compat with scale factor, e.g. size of --crop-box-b (x_min, y_min, z_min, x_max, y_max, z_max) is exactly 8 times bigger than --crop-box-a (x_min, y_min, x_max, y_max) in X and Y dim.

```shell
python src/LF_linearsys/io/preprocess_pair.py --input-dir "data/raw/lightsheet_vol_6.9" --img-dir "data/raw/120um_imgs" --output-dir "data/processed/crop_120um" --downsampling-rate 1.0 --scale-factor 8.0 --crop-box-b "1140,830,1652,1342" --crop-box-a "5,268,18,69,332,82"
```


2. Update config files to use `data_dir` (for pairs) or `points_dir` (for points) instead of `raw_*_dir`.




## Troubleshooting

- **OOM Errors:** Decrease `file_batch_size` (points) or `joint_pair_num` (pairs) in the config.
- **CUDA Errors:** Ensure `device: "cuda"` is set in config and you have a valid GPU.
