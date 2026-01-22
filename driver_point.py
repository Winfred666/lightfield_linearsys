import argparse
import yaml
import torch
from pathlib import Path
import glob
import logging
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.point_system import PointLinearSystem
from src.core.batched_fista import BatchedFISTASolver
from src.core.batched_ista import BatchedISTASolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    cfg = load_config(config_path)
    
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    data_dir = Path(cfg['data']['points_dir'])
    output_dir = Path(cfg['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pattern = cfg['data']['batch_pattern']
    files = sorted(list(data_dir.glob(pattern)))
    
    if not files:
        logger.error(f"No files found matching {pattern} in {data_dir}")
        return

    logger.info(f"Found {len(files)} batch files.")
    
    # Store results to reconstruct full volume
    # We need to know dimensions.
    # We can infer from the first batch and coords.
    # Or just save solved batches and have a separate stitcher.
    # Let's save solved batches first.
    
    for f in files:
        logger.info(f"Processing {f}...")
        data = torch.load(f, map_location='cpu')
        
        # A: (B, Pairs, Z)
        # b: (B, Pairs)
        A_full = data['A']
        b_full = data['b']
        coords_full = data['coords'] # (B, 2) -> (x, y)
        batch_idx = data.get('batch_index', 0)
        
        total_points = A_full.shape[0]
        sub_batch_size = cfg['solver'].get('batch_size', total_points) # Default to full if not set
        
        x_hat_list = []
        
        num_sub_batches = (total_points + sub_batch_size - 1) // sub_batch_size
        logger.info(f"Splitting {total_points} points into {num_sub_batches} sub-batches of size {sub_batch_size}...")
        
        for i in range(num_sub_batches):
            start = i * sub_batch_size
            end = min(start + sub_batch_size, total_points)
            
            # Slice
            A_sub = A_full[start:end]
            b_sub = b_full[start:end]
            
            # Create System
            # Move to GPU inside system init
            threshold_A = cfg['data'].get('threshold_A', 0.20)
            threshold_b = cfg['data'].get('threshold_b', 1.0)
            system = PointLinearSystem(A_sub, b_sub, device=device, threshold_A=threshold_A, threshold_b=threshold_b)

            # Create Solver
            solver_type = cfg['solver'].get('type', 'fista')
            if solver_type == 'fista':
                solver = BatchedFISTASolver(
                    system,
                    lambda_reg=cfg['solver']['lambda_reg'],
                    n_iter=cfg['solver']['n_iter'],
                    output_dir=output_dir,
                    positivity=cfg['solver']['positivity']
                )
            elif solver_type == 'ista':
                solver = BatchedISTASolver(
                    system,
                    lambda_reg=cfg['solver']['lambda_reg'],
                    n_iter=cfg['solver']['n_iter'],
                    output_dir=output_dir,
                    positivity=cfg['solver']['positivity']
                )
            else:
                logger.error(f"Unknown solver type: {solver_type}")
                break
            
            # Solve
            x_hat_sub = solver.solve() # (B_sub, Z)
            
            # Move to CPU immediately to save GPU memory
            x_hat_list.append(x_hat_sub.cpu())
            
            # Clean up
            del system
            del solver
            del x_hat_sub
            torch.cuda.empty_cache()
            
        # Concatenate results
        x_hat = torch.cat(x_hat_list, dim=0)
        
        # Save result
        out_name = f"solved_{f.name}"
        save_path = output_dir / out_name
        
        # Move to CPU for saving
        save_data = {
            'x_hat': x_hat, # Already on CPU
            'coords': coords_full,
            'batch_index': batch_idx
        }
        torch.save(save_data, save_path)
        logger.info(f"Saved {save_path}")

    logger.info("All batches processed.")
    
    # Optional: Reconstruct Volume
    # We can do a quick check of max coord to allocate volume
    # Or just leave it to a visualizer script.
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/solve_point.yaml", help="Path to config")
    args = parser.parse_args()
    
    main(args.config)
