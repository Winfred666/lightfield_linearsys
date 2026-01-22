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

## Current result

### Execute

Running

```bash
python driver.py --config config/default.yaml
```

## TODO

- [X] Check whether the data dimension is correct as `## Raw Data Statement` says.

- [X] Preprocess each image-light_field pairs into a clean Ax=b data set file, sparse storing way is wellcomed if there are many zeros. 

- [X] Implement FISTA solver core logic with sparse/compressed data handling.
- [X] Create driver script to solve joint system of multiple pairs.


- [ ] In `src/io/preprocess_pair.py`, write downsample version of A,b pair to accelerate solving process.

## Develop Log

- **2026-01-18**: Implemented `src/core/solver.py` containing `FISTASolver` and `LinearSystem`.
    - Added memory optimization: converts dense tensors to `float16` and explicitly manages memory to allow processing multiple >7GB pairs on limited RAM.
    - Implemented row filtering (though `pair_1` was found to be 100% dense in pixels).
    - Created `driver.py` to join multiple `Ax=b` systems. Tested with 2 pairs successfully.
    - Added unit tests in `test/test_solver.py`.
    - Note: `float16` usage may cause logging to show `inf` loss due to limited range, but solver runs.
- **2026-01-19**: Refactored codebase and added visualization.
    - **Refactoring**: Moved `LinearSystem` to `src/core/linear_system.py` to decouple it from the solver logic.
    - **Solver**: Enhanced `FISTASolver` to log residual norms ($||Ax-b||$) and sparsity per iteration.
    - **Config**: Introduced `config/default.yaml` to manage solver parameters, data paths, and experimental settings.
    - **Driver**: Updated `driver.py` to use YAML config, output comprehensive results (reconstruction, convergence history, projection validation), and support future solvers.
    - **Visualization**: Created `scripts/visualize_result.py` to generate convergence plots, fidelity checks ($Ax$ vs $b$), volume slices, and video scans.

- **2026-01-20**: use ISTA, try point-wise solver.

- **2026-01-22**: use H5 to speedup executing.