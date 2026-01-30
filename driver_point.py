import argparse
import yaml
import torch
import torch.multiprocessing as mp
from pathlib import Path
import logging
import logging.handlers
import sys
import os
import time
from datetime import datetime
import gc

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LF_linearsys.core.point_system import PointLinearSystem
from LF_linearsys.core.batched_newton_activeset import BatchedRegNewtonASSolver
from LF_linearsys.core.batched_ista import BatchedISTASolver
from LF_linearsys.utils.volume2mesh import export_volume_to_obj
from LF_linearsys.io.preprocess_point import preprocess_points_from_raw
from LF_linearsys.utils.time_recorder import TimeRecorder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable for workers to access the shared tensor
shared_x_global = None


def _configure_worker_logging(log_queue):
    """Configure logging in a worker process to forward records to the parent."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any inherited handlers (including the parent's FileHandler) to avoid
    # file write races / duplicated logs.
    for h in root.handlers[:]:
        root.removeHandler(h)

    qh = logging.handlers.QueueHandler(log_queue)
    root.addHandler(qh)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_worker(x_global_tensor, log_queue):
    """Initialize worker process by setting the shared global volume and logging."""
    global shared_x_global
    shared_x_global = x_global_tensor
    _configure_worker_logging(log_queue)

def process_file(args):
    """Worker function to process a single points file."""
    file_path, gpu_id, cfg, output_dir = args
    global shared_x_global
    
    # Initialize timing recorder for this worker
    timer = TimeRecorder(output_dir, worker_id=f"gpu_{gpu_id}")
    
    try:
        # Assign device
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
        logger.info(f"START {file_path.name} on {device}")

        # Start IO timing
        timer.start('io')

        # Load data: IO time is the main bottleneck !!!
        data = torch.load(file_path, map_location='cpu')
        
        A_full = data['A']
        b_full = data['b']
        coords_full = data['coords'] # (N, 2)
        
        # Stop IO timing, start overhead for setup
        timer.stop()
        timer.start('overhead')
        
        total_points = A_full.shape[0]
        # Solver batch size (sub-batch within the file to fit in GPU memory)
        sub_batch_size = cfg['solver'].get('batch_size', total_points)
        
        num_sub_batches = (total_points + sub_batch_size - 1) // sub_batch_size
        
        # logger.info(f"Processing {file_path.name} on {device} ({num_sub_batches} sub-batches)")
        
        for i in range(num_sub_batches):
            start = i * sub_batch_size
            end = min(start + sub_batch_size, total_points)
            
            A_sub = A_full[start:end]
            b_sub = b_full[start:end]
            coords_sub = coords_full[start:end]
            
            # Create System
            threshold_A = cfg['data'].get('threshold_A', 0.20)
            threshold_b = cfg['data'].get('threshold_b', 1.0)
            
            # Initialize system (data moves to device inside class usually, or we pass device)
            system = PointLinearSystem(A_sub, b_sub, device=device, threshold_A=threshold_A, threshold_b=threshold_b)

            # Create Solver
            solver_type = cfg['solver'].get('type', 'fista')
            
            common_params = {
                'lambda_reg': cfg['solver']['lambda_reg'],
                'n_iter': cfg['solver']['n_iter'],
                'output_dir': output_dir,
                'positivity': cfg['solver']['positivity']
            }

            if solver_type == 'newton':
                solver = BatchedRegNewtonASSolver(system, **common_params)
            elif solver_type == 'ista':
                solver = BatchedISTASolver(system, **common_params)
            else:
                raise ValueError(f"Unknown solver type: {solver_type}")
            
            # Stop overhead timing, start compute timing
            timer.stop()
            timer.start('compute')
            
            # Solve
            # Only plot / record loss curve periodically to reduce overhead.
            # Pass tag=None to disable history/plot inside the solver.
            if (i % 5) == 0:
                solve_tag = f"{file_path.stem}_b{i:04d}"
            else:
                solve_tag = None
            x_hat_sub = solver.solve(tag=solve_tag) # Expected (B_sub, Z)
            
            # Stop compute timing, start overhead timing for cleanup/update
            timer.stop()
            timer.start('overhead')
            
            # Move result to CPU
            x_hat_cpu = x_hat_sub.detach().cpu()
            
            # Update global volume in shared memory
            # coords_sub is (B_sub, 2) -> (x, y)
            # We assume unique coordinates across batches as stated by user.
            shared_x_global[coords_sub[:, 0], coords_sub[:, 1], :] = x_hat_cpu
            
            # --- hard cleanup ---
            del x_hat_cpu, x_hat_sub
            del solver
            del system
            del A_sub, b_sub, coords_sub

            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Finished batch {i + 1}/{num_sub_batches} of {file_path.name} on {device}")

        # Stop final overhead timing
        timer.stop()
        
        # Save timing results
        timer.save()
        summary = timer.get_summary()
        logger.info(f"Worker {gpu_id} ({file_path.name}) Timing: IO={summary['io_time']:.2f}s, Compute={summary['compute_time']:.2f}s, Overhead={summary['overhead_time']:.2f}s")

        return f"Finished {file_path.name} on {device}"
        
    except Exception as e:
        logger.exception(f"Fatal error processing {file_path}")
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/solve_point.yaml", help="Path to config")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes. Default: min(files, GPUs * 4)")
    args = parser.parse_args()
    
    # Use spawn for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    cfg = load_config(args.config)

    # Optional mesh export config (mirrors driver_pair.py style)
    # Can be a single float/int or a list/tuple of iso values.
    mesh_isos = cfg.get('mesh_iso', None)

    # Timestamped output folder + file logging, mirroring driver_pair.py
    output_dir = Path(cfg['data']['output_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / "run.log"

    # --- Multiprocessing-safe logging ---
    # All workers send LogRecords to this queue; the parent writes them to file.
    log_queue = mp.Manager().Queue(-1)

    # Parent-side file handler + listener
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    queue_listener = logging.handlers.QueueListener(log_queue, file_handler)
    queue_listener.start()

    # Parent root logger: keep console output, and also forward to file via queue.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))

    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    # --- File Discovery (Raw vs Processed) ---
    raw_A_dir = cfg['data'].get('raw_A_dir')
    if raw_A_dir:
        # RAW Mode
        logger.info("Configuration points to raw data. Generating points batches on-the-fly.")
        
        # Default batch size for files (not to be confused with solver batch_size)
        file_batch_size = cfg['data'].get('file_batch_size', 30000)

        # Determine where to save the points
        # If points_dir is specified (even if commented out in user config, we assume the user might have uncommented it or we check if it exists in the dict),
        # use it. Otherwise, create a default in data/points_raw_...
        points_dest = cfg['data'].get('points_dir')
        if points_dest:
             cache_dir = Path(points_dest)
        else:
             cache_dir = Path(f"data/points_raw_{file_batch_size}")
        
        logger.info(f"Points will be cached/saved to: {cache_dir}")
        
        # files = preprocess_points_from_raw(...) uses this output_dir
        
        files = preprocess_points_from_raw(
            input_dir=raw_A_dir,
            img_dir=cfg['data']['raw_b_dir'],
            output_dir=cache_dir,
            downsampling_rate=cfg['data']['downsampling_rate'],
            scale_factor=cfg['data'].get('scale_factor', 8.0),
            max_pairs=cfg['data'].get('max_pairs'),
            batch_size=file_batch_size
        )
        logger.info(f"Generated {len(files)} batch files in {cache_dir}")
        
    else:
        # Processed Mode (Legacy)
        data_dir = Path(cfg['data']['points_dir'])
        pattern = cfg['data']['batch_pattern']
        files = sorted(list(data_dir.glob(pattern)))
        logger.info(f"Found {len(files)} batch files in {data_dir} matching {pattern}")
    
    if not files:
        logger.error(f"No files found to process.")
        return

    # Determine global volume dimensions
    try:
        # Peek at the first file for dimensions
        sample = torch.load(files[0], map_location='cpu')
        if 'full_dims' in sample:
            X, Y, Z = sample['full_dims']
            logger.info(f"Detected dimensions from file: {X}, {Y}, {Z}")
        else:
            logger.warning("full_dims not found in file. Falling back to hardcoded default (1192, 2048, 800).")
            X, Y, Z = 1192, 2048, 800
    except Exception as e:
        logger.error(f"Failed to read dimensions from first file: {e}")
        return

    logger.info(f"Allocating shared global volume ({X}, {Y}, {Z})...")
    # Allocate tensor in shared memory
    x_global = torch.zeros((X, Y, Z), dtype=torch.float32)
    x_global.share_memory_()
    
    # GPU setup
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No GPUs found! Running on CPU.")
        num_gpus = 1 # Logical count for modulo ops
        using_cuda = False
    else:
        logger.info(f"Available GPUs: {num_gpus}")
        using_cuda = True
    
    # Worker setup
    if args.num_workers is None:
        # Default strategy: 1 workers per GPU (if CUDA), else just 1 workers total or len(files)
        workers_per_gpu = 1 if using_cuda else 1
        num_workers = min(len(files), num_gpus * workers_per_gpu)
    else:
        num_workers = args.num_workers
        
    logger.info(f"Starting execution with {num_workers} workers.")
    
    # Prepare tasks: (file, gpu_id, config)
    tasks = []
    for i, f in enumerate(files):
        # Round-robin assignment of GPUs
        gpu_id = i % num_gpus
        tasks.append((f, gpu_id, cfg, output_dir))
        
    t0 = time.time()
    
    try:
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(x_global, log_queue)) as pool:
            # Map tasks to workers
            results = pool.map(process_file, tasks, chunksize=1)
    except BaseException:
        logger.exception("Fatal error in worker pool; terminating.")
        try:
            queue_listener.stop()
        except Exception:
            pass
        sys.exit(1)
    finally:
        # Ensure all pending log records are flushed to disk.
        try:
            queue_listener.stop()
        except Exception:
            pass
        
    # Check results
    errors = [r for r in results if "Error" in r]
    if errors:
        logger.error(f"Encountered {len(errors)} errors:")
        for e in errors:
            logger.error(e)
    else:
        logger.info("All tasks completed successfully.")
            
    logger.info(f"Total processing time: {time.time() - t0:.2f}s")
    
    # Save final result
    logger.info("Saving reconstructed volume...")
    save_path = output_dir / "reconstruction.pt"
    torch.save(
        {
            'reconstruction': x_global, # This is the shared tensor, now populated
            'X': X, 'Y': Y, 'Z': Z,
            'source_batches': [str(p) for p in files],
        },
        save_path,
    )
    logger.info(f"Saved {save_path}")

    # Visualization (Always enabled, stride=5)
    try:
        from LF_linearsys.utils.visualize_slices import visualize_reconstruction_and_reprojection

        out_dir = output_dir / "viz"

        logger.info("Starting mandatory visualization (stride=5)...")
        visualize_reconstruction_and_reprojection(
            vol=x_global.detach().cpu().float(),
            output_dir=out_dir,
            threshold_A=float(cfg["data"].get("threshold_A", 0.1) or 0.1),
            data_dir=cfg["data"].get("data_dir"),
            raw_A_dir=cfg["data"].get("raw_A_dir"),
            raw_b_dir=cfg["data"].get("raw_b_dir"),
            downsampling_rate=float(cfg["data"].get("downsampling_rate", 0.125) or 0.125),
            scale_factor=float(cfg["data"].get("scale_factor", 8.0) or 8.0),
            stride_pairs=5,
            make_z_scan_video=True,
        )
        logger.info(f"Saved visualizations to {out_dir}")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # Optional: export snapshot + mesh (similar to driver_pair.py)
    try:
        if mesh_isos is not None:
            if isinstance(mesh_isos, (list, tuple)):
                iso_list = [float(v) for v in mesh_isos]
            else:
                iso_list = [float(mesh_isos)]

            vol_np = x_global.cpu().float().numpy()
            for iso in iso_list:
                logger.info(f"Exporting mesh with iso={iso} ...")
                # One mesh per iso to avoid overwriting.
                obj_path = output_dir / f"reconstruction_iso_{iso}.obj"
                export_volume_to_obj(
                    vol_np,
                    obj_path,
                    iso_value=iso,
                )
                logger.info(f"Saved mesh to {obj_path}")
    except Exception as e:
        logger.warning(f"Mesh export failed: {e}")

    # Persist config for reproducibility (like driver_pair.py)
    try:
        with open(output_dir / "config_used.yaml", 'w') as f:
            yaml.safe_dump(cfg, f)
    except Exception as e:
        logger.warning(f"Failed to write config_used.yaml: {e}")

if __name__ == "__main__":
    main()