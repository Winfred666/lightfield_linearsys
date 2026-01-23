import argparse
import torch
import glob
import os
import logging
import yaml
import h5py
import re
from pathlib import Path
from datetime import datetime
from src.io.data_postclean import compute_active_z_range
from src.core.linear_system_pair import LinearSystemPair
from src.core.masked_system_pair import LinearSystemPairMasked
from src.core.fista import FISTASolver
from src.core.ista import ISTASolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Driver for Light Field Solver")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Extract config
    data_dir = cfg['data']['data_dir']
    output_dir = cfg['data']['output_dir']
    init_global_x_path = cfg['data'].get('init_global_x')
    max_pairs = cfg['data'].get('max_pairs')
    threshold_A = float(cfg['data'].get('threshold_A', 1e-3))
    threshold_b = float(cfg['data'].get('threshold_b', 1.0))

    masking_cfg = cfg.get('masking', {}) or {}
    masking_enabled = bool(masking_cfg.get('enabled', False))
    assume_non_negative_A = bool(masking_cfg.get('assume_non_negative_A', True))
    
    solver_cfg = cfg['solver']
    solver_type = solver_cfg.get('type', 'fista')
    device = cfg.get('device', 'cpu')
    
    if torch.cuda.is_available() and device == 'cuda':
        device = 'cuda'
        logger.info("Using CUDA")
    else:
        device = 'cpu'
        logger.info("Using CPU")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir_ts = str(Path(output_dir) / timestamp)
    Path(output_dir_ts).mkdir(parents=True, exist_ok=True)
    
    log_file_path = Path(output_dir_ts) / "run.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Output directory: {output_dir_ts}")
    
    # 1. Find files
    pattern = os.path.join(data_dir, "pair_*.h5")
    # Natural sort
    files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'pair_(\d+).h5', x).group(1)))
    
    if not files:
        logger.error(f"No files found matching {pattern}")
        return
        
    if max_pairs:
        logger.info(f"Limiting to {max_pairs} pairs (from config)")
        files = files[:max_pairs]
    else:
        logger.info(f"Processing all {len(files)} pairs.")
        
    logger.info(f"Found {len(files)} pair files.")
    
    # Global volume state, Try Initialize global x
    x_global = None
    if init_global_x_path and os.path.exists(init_global_x_path):
        logger.info(f"Loading initial global volume from {init_global_x_path}")
        loaded = torch.load(init_global_x_path, map_location='cpu')
        if isinstance(loaded, dict) and 'reconstruction' in loaded:
            x_global = loaded['reconstruction']
        elif isinstance(loaded, torch.Tensor):
            x_global = loaded
        else:
            logger.warning(f"Could not understand format of {init_global_x_path}, initializing with zeros.")
        
        # Ensure float16
        x_global = x_global.to(torch.float16)
    elif init_global_x_path:
        logger.warning(f"init_global_x path {init_global_x_path} not found.")

    # 2. --------Iterative Solve--------
    for i, f in enumerate(files):
        logger.info(f"=== Processing pair {i+1}/{len(files)}: {f} ===")
        try:
            # Load HDF5
            with h5py.File(f, 'r') as hf:
                 # Read into memory
                 A_np = hf['A'][:]
                 b_np = hf['b'][:]
                 
            # Convert to torch
            A = torch.from_numpy(A_np)
            b = torch.from_numpy(b_np)

            # Check shape
            if x_global is None or x_global.shape != A.shape:
                if x_global is not None:
                    logger.warning(f"Shape mismatch: Loaded {x_global.shape}, Expected {A.shape}. Resizing/Resetting to zeros.")
                X, Y, Z = A.shape
                logger.info(f"Initializing Zero global volume with shape ({X}, {Y}, {Z})")
                x_global = torch.zeros(X, Y, Z, dtype=torch.float16, device='cpu')
            
            # Move to device if small enough, but let system handle device placement
            # Ideally keep on CPU until sliced
            
            if A is None or b is None:
                logger.warning(f"Invalid data in {f}, skipping.")
                continue
            
            # --- Z-Slicing Optimization ---
            # Compute active Z range for this pair
            z_min, z_max = compute_active_z_range(A, threshold_A=threshold_A)
            
            if z_min >= z_max:
                logger.warning("Active Z range empty. Skipping pair.")
                continue
                
            # Slice A: (X, Y, Z) -> (X, Y, z_range)
            A_sub = A[:, :, z_min:z_max]
            
            # Check if sliced A is valid
            if A_sub.numel() == 0:
                 logger.warning("Sliced A is empty. Skipping.")
                 continue

            logger.info("Setting up Linear System for slice...")
            A_list = [A_sub]
            b_list = [b]
            
            if masking_enabled:
                system = LinearSystemPairMasked(
                    A_list,
                    b_list,
                    device=device,
                    threshold_A=threshold_A,
                    assume_non_negative_A=assume_non_negative_A,
                    threshold_b=threshold_b,
                )
            else:
                system = LinearSystemPair(A_list, b_list, device=device, threshold_A=threshold_A, threshold_b=threshold_b)
            
            if system.valid_indices.numel() == 0:
                logger.warning("No valid equations in this pair after filtering. Skipping solve.")
                del system, A_list, b_list, A, b
                import gc
                gc.collect()
                continue
            
            logger.info(f"System ready. Valid equations: {system.valid_indices.numel()}")

            if solver_type == 'fista':
                solver = FISTASolver(
                    system,
                    output_dir=Path(output_dir_ts),
                    tag=f"pair_{i+1}",
                    lambda_reg=solver_cfg.get('lambda_reg', 0.0),
                    lipchitz=solver_cfg.get('lipchitz', 1.0)
                )
            elif solver_type == 'ista':
                solver = ISTASolver(
                    system,
                    output_dir=Path(output_dir_ts),
                    tag=f"pair_{i+1}",
                    lambda_reg=solver_cfg.get('lambda_reg', 0.0),
                    lipchitz=solver_cfg.get('lipchitz', 1.0)
                )
            else:
                logger.error(f"Unknown solver type: {solver_type}")
                break
                
            # Prepare initial guess from global volume (sliced)
            x0_sub = x_global[:, :, z_min:z_max].to(device)
            
            # Solve
            logger.info("Starting Solver for current pair (slice)...")
            x_result_sub = solver.solve(x0_sub, n_iter=solver_cfg.get('n_iter', 100))
            
            # Update Global Volume (Slice + valid_indices only)
            logger.info("Updating global volume (max aggregation)...")
            x_result_sub_cpu = x_result_sub.cpu()
            
            if x_result_sub_cpu.shape != x_global[:, :, z_min:z_max].shape:
                 logger.error(f"Shape mismatch: Global slice {x_global[:, :, z_min:z_max].shape}, Result {x_result_sub_cpu.shape}")
                 raise ValueError("Shape mismatch during global volume update.")

            # Update only the active slice
            current_slice = x_global[:, :, z_min:z_max]
            x_global[:, :, z_min:z_max] = torch.max(current_slice, x_result_sub_cpu)


            # visualize result after some joint pairs opt.
            if i > 0 and i % 10 == 0:
                mesh_iso = cfg.get('mesh_iso', None)
                if mesh_iso is not None:
                    mesh_iso = float(mesh_iso)
                    solver._export_mesh(x_global, mesh_iso)
                solver._export_volume_pt(x_global)
            # Cleanup
            del system, solver, x_result_sub, A_list, b_list, A_sub, x_result_sub_cpu, current_slice
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process {f}: {e}", exc_info=True)
            continue

    save_path = Path(output_dir_ts) / "reconstruction.pt"
    
    logger.info(f"Saving final result to {save_path}")
    torch.save({"reconstruction": x_global}, save_path)
    
    with open(Path(output_dir_ts) / "config_used.yaml", 'w') as f:
        yaml.dump(cfg, f)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()
