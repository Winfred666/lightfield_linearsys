#!/usr/bin/env python3
"""Run visualize_density_slices.py for all MLP reconstruction .pt files."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Mapping of reconstruction files to their corresponding data directories
RECONSTRUCTIONS = [
    {
        "pt_file": "/home/ym.xiao/workspace/lightfield_linearsys/result/MLP_result/pt_files/reconstruction20um.pt",
        "data_dir": "/home/ym.xiao/workspace/lightfield_linearsys/data/processed/crop_20um/ds1"
    },
    {
        "pt_file": "/home/ym.xiao/workspace/lightfield_linearsys/result/MLP_result/pt_files/reconstruction60um.pt",
        "data_dir": "/home/ym.xiao/workspace/lightfield_linearsys/data/processed/crop_60um/ds0p5"
    },
    {
        "pt_file": "/home/ym.xiao/workspace/lightfield_linearsys/result/MLP_result/pt_files/reconstruction80um.pt",
        "data_dir": "/home/ym.xiao/workspace/lightfield_linearsys/data/processed/crop_80um/ds0p25"
    },
    {
        "pt_file": "/home/ym.xiao/workspace/lightfield_linearsys/result/MLP_result/pt_files/reconstruction120um.pt",
        "data_dir": "/home/ym.xiao/workspace/lightfield_linearsys/data/processed/crop_120um/ds0p166667"
    }
]

def run_visualization(pt_file: str, data_dir: str):
    """Run visualize_density_slices.py for a single reconstruction."""
    print(f"\n{'='*80}")
    print(f"Running visualization for: {Path(pt_file).name}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*80}")

    # Create output directory name based on input file
    output_dir = Path(pt_file).parent / "viz" / Path(pt_file).stem

    # Build command
    cmd = [
        "python", "scripts/visualize_density_slices.py",
        pt_file,
        "--data-dir", data_dir,
        "--output-dir", str(output_dir)
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        # Run the command (subprocess itself, so we can safely parallelize the Python wrapper).
        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "pt_file": pt_file,
            "data_dir": data_dir,
            "output_dir": str(output_dir),
            "cmd": cmd,
            "returncode": int(result.returncode),
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except Exception as e:
        return {
            "pt_file": pt_file,
            "data_dir": data_dir,
            "output_dir": str(output_dir),
            "cmd": cmd,
            "returncode": 1,
            "stdout": "",
            "stderr": f"Error running visualization: {e}",
        }

def main():
    parser = argparse.ArgumentParser(description="Run visualize_density_slices.py for all configured reconstructions.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(4, (os.cpu_count() or 1)),
        help="Maximum parallel visualizations to run at once (default: min(4, cpu_count)).",
    )
    args = parser.parse_args()

    print("Starting visualization of all MLP reconstructions")
    print(f"Total reconstructions: {len(RECONSTRUCTIONS)}")
    print(f"Parallel workers: {args.max_workers}")

    jobs: list[tuple[str, str]] = []
    for recon in RECONSTRUCTIONS:
        pt_path = Path(recon["pt_file"])
        data_dir = Path(recon["data_dir"])

        # Check if files exist
        if not pt_path.exists():
            print(f"Error: PT file not found: {pt_path}")
            continue

        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            continue

        jobs.append((str(pt_path), str(data_dir)))

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = [ex.submit(run_visualization, pt_file, data_dir) for pt_file, data_dir in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

            # Print each job result as it completes (avoids interleaved live output).
            print(f"\n{'='*80}")
            print(f"Completed: {Path(res['pt_file']).name}")
            print(f"Data directory: {res['data_dir']}")
            print(f"Output directory: {res['output_dir']}")
            print(f"Command: {' '.join(res['cmd'])}")
            if res["stdout"]:
                print("STDOUT:")
                print(res["stdout"])
            if res["stderr"]:
                print("STDERR:")
                print(res["stderr"])
            print(f"Return code: {res['returncode']}")
            if res["returncode"] == 0:
                print(f"✓ Successfully created visualizations in: {res['output_dir']}")
            else:
                print(f"✗ Visualization failed for {Path(res['pt_file']).name}")

    n_ok = sum(1 for r in results if r["returncode"] == 0)
    n_fail = len(results) - n_ok
    print("\n" + "=" * 80)
    print(f"All visualizations completed: {n_ok} succeeded, {n_fail} failed")

if __name__ == "__main__":
    main()