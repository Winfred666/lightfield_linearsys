---
name: run_point_solver
description: Configure a solver YAML and run `driver_point.py` on a remote GPU node, then validate results with metrics.
---

# Configure and Run Solver

## Overview

Use this skill when you have a directory of measurements (2D images) and want to reconstruct a 3D volume using the point-based solver entrypoint `driver_point.py`.

This skill covers:

* creating a config YAML that points to your measurement data and output/cache paths
* running the solver on remote GPU nodes
* verifying success by locating outputs and optionally computing a reconstruction metric against a GT volume

## Instructions

Search data/run*.sh to see whether there are `driver_point.py` calling, if so, use them as reference. We prefer parallel execute commands below for different measurement cases for a fast turnaround.

1. **Create a solver config YAML** (for example, `config/solve_my_experiment.yaml`).

	Include (at minimum) these fields:

	* `data.raw_b_dir`: path to the measurement images directory
	* `data.points_dir`: path to cache/store processed point clouds
	* `data.crop_box_b`: crop for the images in `[x0, y0, x1, y1]`
	* `data.crop_box_A`: crop for the 3D volume in `[x0, y0, z0, x1, y1, z1]`
	* `solver.type`: solver backend (for example `newton`, `ista`, etc.)

If the YAML already exist and from the history you know that there is change in raw_b_dir after last run, or you have an uncomplish points batch, delete `points_dir` before running `driver_point.py` again. This ensures that the newly generated points batch will be consistent and complete (e.g., as many files as a reference case).

2. **Decide which machine/GPU(s) to run on**, see skill `use_gpu_node`

3. **Run the solver** via `driver_point.py`.

4. **Check outputs**.

	A successful run should produce a reconstruction artifact under directory `output_dir`/latest_date (often a `reconstruction.pt`). If you want to monitor logs, monitor `run.log` file inside `output_dir`/latest_date.

5. **(Optional) Compute a metric vs GT**.

	If you have a ground-truth `.pt` volume, compute a quantitative metric using `scripts/compute_3d_metric.py`.

### What success looks like

You should consider this skill successful when:

* `driver_point.py` finishes without error, and
* the configured output folder under `result/` contains a **reconstructed volume** (commonly `reconstruction.pt`), and
* (optional) your **metric script runs** and reports reasonable values.

## Examples

### Example: run on a remote GPU node

**SSH** to a GPU host, **activate the environment**, choose GPUs via `CUDA_VISIBLE_DEVICES`, then **run the solver**:

```bash
PYTHON="/data/volume3/share_storage/ym.xiao/miniconda/envs/torch/bin/python"
ssh gpu04 "cd /home/ym.xiao/workspace/lightfield_linearsys && \
CUDA_VISIBLE_DEVICES=0,1 $PYTHON driver_point.py --config config/solve_my_experiment.yaml'"
```

### Example: compute metrics against ground truth

**Run the metric script** on the produced reconstruction:

```bash
$PYTHON scripts/compute_3d_metric.py result/.../reconstruction.pt --gt-path data/.../gt.pt
```

