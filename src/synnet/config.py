"""Central place for all configuration, paths, and parameter."""
import multiprocessing
import os

# Multiprocessing
DEFAULT_MAX_PROCESSES = 31
MAX_PROCESSES = min(
    int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_MAX_PROCESSES)), multiprocessing.cpu_count()
)
