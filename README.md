# FISTA solver for massive low-rank sparse linear system 

## Problem Statement

In this project we are solving a "massive linear system" derived from Structured Illumination Microscopy (SIM) or Light Field Microscopy (LFM) data.

$$g(u, v) = \int_{0}^{Z_{max}} \Phi(u,v,z) \cdot f(u, v, z) \, dz$$

When discreted, it could be written as

$$y(u_i, v_i)= \sum_{z=1}^{N_z} \Phi(u_i, v_i, z) \cdot x(u_i, v_i, z)$$

Here $x$ is voxel denstiy, which is our target. A general way to solve this massive linear system is using a least squares method, and adding regularization terms $\mathcal{R}$ to mitigate ill-posed, also adding constrained $\mathbf{x} > 0$ into the target function.

$$\hat{\mathbf{x}} = \arg \min_{\mathbf{x}} \frac{1}{2} \| A\mathbf{x} - \mathbf{y} \|_2^2 + \lambda \mathcal{R}(\mathbf{x}) + \iota_{\geq 0}(\mathbf{x})$$

The baseline could be FISTA, where we performed gradient descent with momentum $\mathbf{z}_k = \mathbf{x}_k - \alpha A^T (A \mathbf{x}_k - \mathbf{y})$, do proximal projection to get $\mathbf{x}_{k+1} = \text{prox}_{\lambda \mathcal{R}} (\mathbf{z}_k)$, and then update the momentum $\mathbf{y}_{k+1} = \mathbf{x}_{k+1} + \frac{t_k - 1}{t_{k+1}} (\mathbf{x}_{k+1} - \mathbf{x}_k)$. But improvement to this specific problem is still under exploration.


## Raw Data Statement

2. $\Phi_{(u,v)}(z)$ represents the "light field" intensity profile along the ray corresponding to pixel $(u, v)$, which is stored in `data/lightsheet_vol_6.9/Interp_Vol_ID_140.pt`, ... to `/home/ym.xiao/workspace/lightfield_linearsys/data/lightsheet_vol_6.9/Interp_Vol_ID_260.pt`, totally 121 volumes representing different lightning of the same scene $x$.

Details of optical light field: each `.pt` file shapes (149, 600, 100). 149 is the width along x, 600 is the height along y, and 100 is depth along z, so all format should obey x-y-z naming system.

Each voxel in one scalar light field represent 27.6um length in real world.

2. The measurement image $g$ is stored in `/home/ym.xiao/workspace/lightfield_linearsys/data/average_imgs/1scan (1).tif` to `/home/ym.xiao/workspace/lightfield_linearsys/data/average_imgs/1scan (121).tif`, where there is 140 index difference between $g$ and $\Phi$.

Each measurement `.tif` image shapes (2448,2048), where 2448 is width along x axis and 2048 is height along y axis.

Also each pixel in one measurement represent 3.45um length in real world.

The measurement is origined at center of x-y plane in light field, which means that when cliping for valid data, we should clip the redundant padding at left and right of measurement images and that at top and bottom of scaled light-field.

## Execution Workflow

### 1. Preprocessing
First, convert raw data into HDF5 pairs, then optionally into point batches for faster parallel solving.

```bash
# Preprocess pairs (generates data/processed/ds0p125/pair_*.h5)
python src/io/preprocess_pair.py --downsampling-rate 0.125

# Preprocess points (generates data/points_scale_ds0p125/points_batch_*.pt)
python src/io/preprocess_point.py --data_dir data/processed/ds0p125 --batch_size 10000
```

### 2. Solving
Choose between pair-based iteration or point-based parallel solving.

#### Point-based Solver (Recommended for GPU clusters)
Uses multiprocessing to distribute point batches across all available GPUs.
```bash
python driver_point.py --config config/solve_point_ista.yaml
```

#### Pair-based Solver
```bash
python driver_pair.py --config config/solve_pair_ista.yaml
```

### 3. Visualization
```bash
# Visualize processed HDF5 pairs
python scripts/visualize_processed.py data/processed/ds0p125/pair_2.h5

# Interactive 3D visualization of reconstruction
python scripts/visualize_light_field.py result/solve_point_fista/TIMESTAMP/reconstruction.pt
```

## TODO

- [X] Check whether the data dimension is correct as `## Raw Data Statement` says.
- [X] Preprocess each image-light_field pairs into HDF5 Ax=b dataset.
- [X] Implement point-based solver with multiprocessing and shared memory.
- [X] Add interactive 3D visualization with PyVista.
- [ ] Implement automated hyperparameter tuning for $\lambda$ regularization.
- [ ] Add support for 3D TV (Total Variation) regularization.

## Develop Log

- **2026-01-24**: Remove Fista solver, Tried Lawson-Hanson Non-Negative Least Squares (NNLS) solver. But impractical because too slow. Even worse, scipy.optimize.nnls requires to explicitly build A, not matrix-free; For point-wise solver, implement BatchedProjectRegNewtonSolver which help converge faster and let density bunch together along z axis.     
    - Implemented `LawsonHansonNNLSSolver`, deleted.
    - Integrated NNLS into `driver_pair.py` as a selectable solver type (deprecated)



- **2026-01-23**: Major performance and scalability update.
    - Switched to HDF5 for pair storage to allow partial slicing.
    - Implemented `driver_point.py` with `torch.multiprocessing` and shared memory for massive parallelization across 8 GPUs.
    - Added `full_dims` metadata to point batches to automate reconstruction volume allocation.
    - Enhanced `visualize_light_field.py` with interactive clipping and thresholding widgets.

- **2026-01-22**: use H5 to speedup executing.

- **2026-01-20**: use ISTA, try point-wise solver.

- **2026-01-19**: Refactored codebase and added visualization.
    - **Refactoring**: Moved `LinearSystem` to `src/core/linear_system.py` to decouple it from the solver logic.
    - **Solver**: Enhanced `FISTASolver` to log residual norms ($||Ax-b||$) and sparsity per iteration.
    - **Config**: Introduced `config/default.yaml` to manage solver parameters, data paths, and experimental settings.
    - **Driver**: Updated `driver.py` to use YAML config, output comprehensive results (reconstruction, convergence history, projection validation), and support future solvers.
    - **Visualization**: Created `scripts/visualize_result.py` to generate convergence plots, fidelity checks ($Ax$ vs $b$), volume slices, and video scans.

- **2026-01-18**: Implemented `src/core/solver.py` containing `FISTASolver` and `LinearSystem`.
    - Added memory optimization: converts dense tensors to `float16` and explicitly manages memory to allow processing multiple >7GB pairs on limited RAM.
    - Implemented row filtering.
    - Created `driver.py` to join multiple `Ax=b` systems.
    - Added unit tests in `test/test_solver.py`.