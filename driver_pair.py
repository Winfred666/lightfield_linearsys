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
from src.io.data_postclean import compute_valid_z_indices
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
    joint_pair_num = int(cfg['data'].get('joint_pair_num', 1))
    if joint_pair_num < 1:
        logger.warning(f"joint_pair_num={joint_pair_num} invalid, forcing to 1")
        joint_pair_num = 1
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
    n_files = len(files)
    n_batches = (n_files + joint_pair_num - 1) // joint_pair_num
    for batch_idx in range(n_batches):
        start = batch_idx * joint_pair_num
        end = min(start + joint_pair_num, n_files)
        batch_files = files[start:end]
        logger.info(
            f"=== Processing batch {batch_idx+1}/{n_batches}: pairs {start+1}-{end} (joint_pair_num={joint_pair_num}) ==="
        )

        if not batch_files:
            continue

        try:
            # Load batch HDF5 -> lists of A,b
            A_full_list = []
            b_list = []
            X = Y = Z = None
            for f in batch_files:
                with h5py.File(f, 'r') as hf:
                    A_np = hf['A'][:]
                    b_np = hf['b'][:]

                A = torch.from_numpy(A_np)
                b = torch.from_numpy(b_np)

                if A is None or b is None:
                    logger.warning(f"Invalid data in {f}, skipping it.")
                    continue

                if X is None:
                    X, Y, Z = A.shape
                elif (X, Y, Z) != tuple(A.shape):
                    raise ValueError(f"Shape mismatch in batch: expected {(X, Y, Z)}, got {tuple(A.shape)} for {f}")

                A_full_list.append(A)
                b_list.append(b)

            if not A_full_list:
                logger.warning("No valid pairs in this batch; skipping.")
                continue

            # Check/init global volume shape
            if x_global is None or x_global.shape != A_full_list[0].shape:
                if x_global is not None:
                    logger.warning(
                        f"Shape mismatch: Loaded {x_global.shape}, Expected {A_full_list[0].shape}. Resizing/Resetting to zeros."
                    )
                X, Y, Z = A_full_list[0].shape
                logger.info(f"Initializing Zero global volume with shape ({X}, {Y}, {Z})")
                x_global = torch.zeros(X, Y, Z, dtype=torch.float16, device='cpu')

            # --- Z-index selection (union across batch) ---
            valid_z_indices = compute_valid_z_indices(A_full_list, threshold_A=threshold_A)
            if valid_z_indices.numel() == 0:
                logger.warning("Valid Z indices empty for this batch. Skipping.")
                continue

            # Slice A by explicit indices: (X,Y,Z)->(X,Y,Zk)
            # NOTE: gather is not needed; advanced indexing is fine and keeps a new tensor.
            A_list = [Ai.index_select(2, valid_z_indices) for Ai in A_full_list]
            Zk = int(valid_z_indices.numel())
            # Sanity
            if any(a.numel() == 0 for a in A_list):
                logger.warning("At least one A_sub is empty after z-index selection; skipping batch.")
                continue

            logger.info(f"Setting up Linear System for Zk={Zk} slices...")
            
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
                    tag=f"batch_{batch_idx+1}",
                    lambda_reg=solver_cfg.get('lambda_reg', 0.0),
                    lipchitz=solver_cfg.get('lipchitz', 1.0)
                )
            elif solver_type == 'ista':
                solver = ISTASolver(
                    system,
                    output_dir=Path(output_dir_ts),
                    tag=f"batch_{batch_idx+1}",
                    lambda_reg=solver_cfg.get('lambda_reg', 0.0),
                    lipchitz=solver_cfg.get('lipchitz', 1.0)
                )
            elif solver_type == 'fista':
                solver = FISTASolver(
                    system,
                    output_dir=Path(output_dir_ts),
                    tag=f"batch_{batch_idx+1}"
                )
            else:
                logger.error(f"Unknown solver type: {solver_type}")
                break
                
            # Prepare initial guess from global volume (selected z indices)
            x0_sub = x_global.index_select(2, valid_z_indices).to(device)
            
            # Solve
            logger.info("Starting Solver for current pair (slice)...")
            x_result_sub = solver.solve(x0_sub, n_iter=solver_cfg.get('n_iter', 100))
            
            # Update Global Volume (Slice + valid_indices only)
            logger.info("Updating global volume (max aggregation)...")
            x_result_sub_cpu = x_result_sub.cpu()

            if x_result_sub_cpu.shape != x0_sub.cpu().shape:
                logger.error(f"Shape mismatch: Global selection {x0_sub.shape}, Result {x_result_sub_cpu.shape}")
                raise ValueError("Shape mismatch during global volume update.")

            # Grab current values at selected Z for comparisons/ghost
            current_sel = x_global.index_select(2, valid_z_indices)

            # --- Ghost voxel debugging ---
            # "Ghost" here means: locations that were previously zero in the global volume slice,
            # but are within this pair's valid equations (A row positive/non-zero after filtering),
            # and the current solve predicts a positive density.
            # We save these newly-created densities into a full-size volume for visualization.
            ghost_eps = float(cfg.get('ghost_eps', 1e-6))
            save_ghost = bool(cfg.get('save_ghost', True))
            ghost_vol = None
            if save_ghost:
                try:
                    # Check lit up area.
                    voxel_mask = torch.zeros((X, Y, Zk), dtype=torch.bool)
                    for A_sub in A_list:
                        voxel_mask |= (A_sub.abs() > threshold_A)

                    # Boolean indexing flattens to 1D; compute ghosts in reduced (X,Y,Zk) space.
                    prev_vals_1d = current_sel[voxel_mask]  # (N_lit_voxels,)
                    new_vals_1d = x_result_sub_cpu[voxel_mask]  # (N_lit_voxels,)

                    # Ghost definition (your latest): global has positive density but current batch solves ~0
                    ghost_mask_1d = (prev_vals_1d > ghost_eps) & (new_vals_1d <= ghost_eps)
                    ghost_delta_1d = torch.where(
                        ghost_mask_1d,
                        prev_vals_1d - new_vals_1d,
                        torch.zeros_like(new_vals_1d),
                    )

                    # Build reduced ghost volume and scatter masked voxels back
                    ghost_reduced = torch.zeros((X, Y, Zk), dtype=torch.float32)
                    ghost_reduced[voxel_mask] = ghost_delta_1d.float()

                    # Scatter into a full-size (X,Y,Z) ghost volume
                    ghost_vol = torch.zeros_like(x_global, dtype=torch.float32)
                    ghost_vol[:, :, valid_z_indices] = ghost_reduced

                    ghost_dir = Path(output_dir_ts) / "ghost_volume"
                    ghost_dir.mkdir(parents=True, exist_ok=True)
                    ghost_path = ghost_dir / f"ghost_batch_{batch_idx+1}.pt"
                    torch.save(
                        {
                            'ghost': ghost_vol,
                            'batch_index': batch_idx + 1,
                            'pair_files': [str(p) for p in batch_files],
                            'valid_z_indices': valid_z_indices,
                            'ghost_eps': ghost_eps,
                        },
                        ghost_path,
                    )
                    logger.info(f"Saved ghost volume to {ghost_path}")
                except Exception as e:
                    logger.warning(f"Ghost export failed for batch {batch_idx+1}: {e}")
            
            
            # Update the global volume at selected z indices with max aggregation
            updated_sel = torch.max(current_sel, x_result_sub_cpu)
            x_global[:, :, valid_z_indices] = updated_sel


            # visualize result after some joint pairs opt.
            if (batch_idx > 0 and batch_idx % 4 == 0) or (batch_idx == n_batches - 1):
                mesh_isos = cfg.get('mesh_iso', None)
                if mesh_isos is not None:
                    if isinstance(mesh_isos, (list, tuple)):
                        iso_list = [float(v) for v in mesh_isos]
                    else:
                        iso_list = [float(mesh_isos)]
                    for iso in iso_list:
                        solver._export_mesh(x_global, iso)
                
                # solver._export_volume_pt(x_global)
            
            
            # Cleanup
            del system, solver, x_result_sub, A_list, b_list, A_full_list, x_result_sub_cpu, current_sel, updated_sel
            if ghost_vol is not None:
                del ghost_vol
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
