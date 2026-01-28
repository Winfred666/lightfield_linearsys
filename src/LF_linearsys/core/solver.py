import torch
import logging
from abc import ABC, abstractmethod
from LF_linearsys.core.linear_system_pair import LinearSystemPair
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from LF_linearsys.utils.volume2mesh import export_volume_to_obj
import time

logger = logging.getLogger(__name__)

class Solver(ABC):
    """
    Abstract base class for solvers.
    """
    def __init__(self,
                 linear_system: LinearSystemPair,
                 output_dir: Path,
                 tag: str = "",
                 **kwargs):
        self.system = linear_system
        # History arrays are appended only at logging steps (every log_interval and
        # the final iter). Keep an explicit iteration index list so plotting can
        # show the true iteration numbers.
        self.history = {'loss': [], 'residual_norm': [], 'sparsity': [], 'iter': []}
        self.output_dir = output_dir
        # Optional suffix for output artifacts (plots/meshes) to avoid overwriting.
        self.tag = str(tag) if tag is not None else ""
        
        self.log_interval = 25

    def _tagged_name(self, base: str, ext: str) -> str:
        """Build output filename with optional tag.

        Example:
          base='loss_curve', ext='.png' -> 'loss_curve_pair_12.png'
        """
        if self.tag.strip() == "":
            return f"{base}{ext}"
        safe = self.tag.strip().replace(" ", "_")
        return f"{base}_{safe}{ext}"

    def solve(self, x0: torch.Tensor, n_iter: int = 100, verbose: bool = True) -> torch.Tensor:
        """
        Runs the solver.
        
        Args:
            x0: Initial guess, shaped (X, Y, Z).
            n_iter: Number of iterations.
            verbose: Print progress.
        """
        x = self._pre_solve(x0)
        
        logger.info(f"Starting solver (n_iter={n_iter})")

        for k in range(n_iter):
            x = self._solve_step(x, k)
            # Logging
            if verbose and (k % self.log_interval == 0 or k == n_iter - 1):
                self.log(x, k)
        x = self._post_solve(x)
        return x

    def log(self, x, k):
        # NOTE: Logging can be surprisingly expensive.
        # Per your preference, we keep computations on the system device (GPU)
        # and only convert final scalars via .item() for Python logging/history.
        t0 = time.time()
        logger.info("Iter %d: computing log statistics...", k)

        with torch.no_grad():
            Ax = self.system.forward(x)
            resid = Ax - self.system.b

            # Compute stats ON DEVICE.
            # Keep these as tensors until the end; calling .item() syncs.
            resid_f = resid.detach().float()
            x_f = x.detach().float()

            resid_norm_t = torch.linalg.vector_norm(resid_f)
            l1_norm_t = torch.linalg.vector_norm(x_f.reshape(-1), ord=1)
            loss_t = 0.5 * (resid_norm_t ** 2)

            # Sparsity = fraction of (near) zeros
            sparsity_t = 1.0 - (x_f > 1e-6).float().mean()

            # Min/max/mean
            min_x_t = x_f.min()
            max_x_t = x_f.max()
            mean_x_t = x_f.mean()

        # Convert to Python scalars for logging/history
        resid_norm = float(resid_norm_t.item())
        l1_norm = float(l1_norm_t.item())
        loss = float(loss_t.item())

        # Exit early if loss becomes NaN/Inf
        if not np.isfinite(loss):
            logger.error(
                "Non-finite loss detected at iter %d: loss=%s, resid_norm=%.4e, l1_norm=%.4e. Stopping solver.",
                k,
                loss,
                resid_norm,
                l1_norm,
            )
            raise FloatingPointError(f"Non-finite loss encountered at iter {k}: {loss}")

        sparsity = float(sparsity_t.item())

        min_x = float(min_x_t.item())
        max_x = float(max_x_t.item())
        mean_x = float(mean_x_t.item())

        self.history['loss'].append(loss)
        self.history['residual_norm'].append(resid_norm)
        self.history['sparsity'].append(sparsity)
        self.history['iter'].append(int(k))
        self.history['min_x'] = min_x  # Store in history for potential future use
        self.history['max_x'] = max_x
        self.history['mean_x'] = mean_x

        dt = time.time() - t0
        logger.info(
            "Iter %d: Loss=%.4e | ||Ax-b||=%.4e | L1=%.4e | Sparsity=%.2f%% | Min=%.4e | Max=%.4e | Mean=%.4e | log_time=%.3fs",
            k,
            loss,
            resid_norm,
            l1_norm,
            sparsity * 100.0,
            min_x,
            max_x,
            mean_x,
            dt,
        )

    def _pre_solve(self, x0: torch.Tensor) -> torch.Tensor:
        """Hook called once before iterations.
        Intended uses:
        - initialize solver state (momentum buffers, step sizes, caches)
        - validate shapes/dtypes
        - optionally create a working copy if the algorithm mutates x in-place
        
        IMPORTANT: Our solvers (currently) create new tensors each step and do not
        mutate x0 in-place, so we can safely return x0 and avoid an expensive clone.
        """
        # Automatically pack x0 if the system supports it (e.g. MaskedLinearSystem)
        if hasattr(self.system, "pack_x"):
            logger.info("System supports pack_x; packing initial guess.")
            return self.system.pack_x(x0)
        return x0

    @abstractmethod
    def _solve_step(self, x: torch.Tensor, k: int) -> torch.Tensor:
        pass

    def _post_solve(self, x: torch.Tensor) -> torch.Tensor:
        # Automatically expand x if the system supports it (e.g. MaskedLinearSystem)
        if hasattr(self.system, "expand_x"):
            logger.info("System supports expand_x; expanding solution.")
            x = self.system.expand_x(x)
            
        logger.info("Solver finished. Running post-processing hooks.")
        self._plot_history()
        # Keep a simple mesh export enabled by default for parity with tests and
        # existing workflows. The driver scripts can do additional exports.
        self._export_mesh(x)
        return x

    def _plot_history(self):
        """Plots the convergence history and saves it to a file."""
        if not self.history['loss']:
            logger.warning("History is empty, skipping plot.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))

        # X axis should reflect the actual solver iteration numbers at which we
        # logged statistics.
        iters = self.history.get('iter')
        if iters and len(iters) == len(self.history['residual_norm']):
            x_axis = iters
        else:
            # Backward-compatible fallback: infer iteration numbers from the log
            x_axis = [i * self.log_interval for i in range(len(self.history['residual_norm']))]

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual Norm')
        ax.plot(x_axis, self.history['residual_norm'], label='Residual Norm')
        ax.set_yscale('log')

        fig.tight_layout()
        plt.title('Solver Convergence')
        
        sub_dir = self.output_dir / "loss_curve"
        sub_dir.mkdir(parents=True, exist_ok=True)
        save_path = sub_dir / self._tagged_name("loss_curve", ".png")
        plt.savefig(save_path)
        logger.info(f"Saved convergence plot to {save_path}")
        plt.close(fig)


    def _export_volume_pt(self, x: torch.Tensor):
        """Exports the reconstructed volume to a .pt file."""
        if x.numel() == 0:
            logger.warning("Reconstruction is empty, skipping volume export.")
            return
        logger.info("Exporting volume to .pt file...")
        volume_output_path = self.output_dir / self._tagged_name("reconstruction", ".pt")
        torch.save(x.cpu(), volume_output_path)
        logger.info(f"Saved reconstructed volume to {volume_output_path}")


    def _export_mesh(self, x: torch.Tensor, iso_value: float = 1.0):
        """Exports the reconstructed volume to an OBJ mesh."""
        if x.numel() == 0:
            logger.warning("Reconstruction is empty, skipping mesh export.")
            return
            
        logger.info("Exporting volume to mesh...")
        sub_dir = self.output_dir / "mesh"
        sub_dir.mkdir(parents=True, exist_ok=True)
        mesh_output_path = sub_dir / self._tagged_name(f"reconstruction_{iso_value}", ".obj")

        volume = x.cpu().float().numpy()
        
        export_volume_to_obj(
            volume,
            mesh_output_path,
            iso_value=iso_value,
        )
