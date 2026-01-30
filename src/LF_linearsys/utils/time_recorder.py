import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TimeRecorder:
    """Records timing for three categories: IO+preprocess, compute/solve, and overhead."""

    def __init__(self, output_dir: Path, worker_id: str = "main"):
        self.output_dir = output_dir
        self.worker_id = worker_id
        self.io_time = 0.0
        self.compute_time = 0.0
        self.overhead_time = 0.0
        self.current_start = None
        self.current_category = None

    def start(self, category: str):
        """Start timing for a specific category."""
        if self.current_start is not None:
            self._record_current()
        self.current_category = category
        self.current_start = time.time()

    def stop(self):
        """Stop timing for the current category."""
        if self.current_start is not None and self.current_category is not None:
            elapsed = time.time() - self.current_start
            if self.current_category == 'io':
                self.io_time += elapsed
            elif self.current_category == 'compute':
                self.compute_time += elapsed
            elif self.current_category == 'overhead':
                self.overhead_time += elapsed
            self.current_start = None
            self.current_category = None

    def _record_current(self):
        """Record the current timing if any."""
        if self.current_start is not None and self.current_category is not None:
            elapsed = time.time() - self.current_start
            if self.current_category == 'io':
                self.io_time += elapsed
            elif self.current_category == 'compute':
                self.compute_time += elapsed
            elif self.current_category == 'overhead':
                self.overhead_time += elapsed
            self.current_start = time.time()  # Restart for new category

    def save(self):
        """Save timing results to a text file."""
        time_record_dir = self.output_dir / "time_record"
        time_record_dir.mkdir(parents=True, exist_ok=True)

        file_path = time_record_dir / f"timing_{self.worker_id}.txt"

        with open(file_path, 'w') as f:
            f.write(f"Timing Results - Worker: {self.worker_id}\n")
            f.write("=" * 50 + "\n")
            f.write(f"IO+Preprocess Time: {self.io_time:.4f} seconds\n")
            f.write(f"Compute/Solve Time: {self.compute_time:.4f} seconds\n")
            f.write(f"Overhead Time: {self.overhead_time:.4f} seconds\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Time: {self.io_time + self.compute_time + self.overhead_time:.4f} seconds\n")

        logger.info(f"Timing results saved to {file_path}")

    def get_summary(self):
        """Return a summary dictionary of timing results."""
        return {
            'worker_id': self.worker_id,
            'io_time': self.io_time,
            'compute_time': self.compute_time,
            'overhead_time': self.overhead_time,
            'total_time': self.io_time + self.compute_time + self.overhead_time
        }
