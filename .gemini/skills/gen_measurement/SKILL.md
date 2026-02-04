---
name: generate_synthetic_measurements
description: Generate synthetic 2D measurement images from a 3D ground-truth volume with a known forward model (system matrices A).
---

# Generate Measurements from Synthetic Data

## Overview

Use this skill when you have a 3D ground-truth (GT) volume and want to simulate what the microscope/system would measure, using precomputed system matrices (the forward operator $A$).

This is typically a prerequisite for solver development and benchmarking:

* GT volume ($x$): `reconstruction.pt` (or similar)
* system matrices ($A$): `Interp_Vol_ID_*.pt` files in a directory
* output measurements ($b = Ax$): saved as `.tif` images

## Instructions

1. **Prepare the ground-truth volume**.

  * Ensure the GT volume is a **`.pt`** file (for example `reconstruction.pt`).
  * If your GT is stored in H5 format, **convert it** using **`scripts/convert_h5_to_pt.py`**.

2. **Locate the forward model (A matrices)**.

  * Identify the directory containing **`Interp_Vol_ID_*.pt`**.
  * Confirm this directory corresponds to the same optical/system setup you want to simulate.

3. **Pick/verify the crop (if needed)**.

  * If the A volume extent is larger than your GT volume (or mismatched), set **`--crop-box-A`**.
  * Format is **`x0 y0 z0 x1 y1 z1`**
  * User would give the crop, or check crop in most related config/*.yaml. If you do not know how to crop, stop and ask user for help.

4. **(Optional) Decide on noise injection**.

  * Use **`--noise-index`** to add noise to a specific measurement index.
  * Use **`--noise-index -1`** for clean measurement without noise.

5. **Generate the measurements**.

  Run **`scripts/generate_bunny_measurements.py`** with:

  * `--gt-vol-path`: path to GT `.pt`
  * `--raw-A-dir`: path to the system matrices
  * `--output-dir`: output folder for `.tif` measurements

### What success looks like

You should consider this skill successful when:

* the script **finishes without error**, and
* `output_dir` contains **one or more `.tif` measurement images**, and
* (recommended) the images look plausible (non-empty, not all zeros/NaNs) when opened or visualized.

## Examples

### Example: generate noise-free measurements with A-cropping

**Generate measurements** from a GT volume using a specified system-matrix directory, and **crop A** to match the GT:

```bash
python scripts/generate_bunny_measurements.py \
  --gt-vol-path data/synthetic/balls/case_1/reconstruction.pt \
  --raw-A-dir data/raw/lightsheet_vol_6.9 \
  --output-dir data/synthetic/balls/case_1/measurements \
  --crop-box-A 5 268 18 69 332 82 \
  --noise-index -1
```

