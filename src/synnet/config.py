"""Central place for all configuration, paths, and parameter."""
import multiprocessing
import os

# Multiprocessing
DEFAULT_MAX_PROCESSES = 31
MAX_PROCESSES = min(
    int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_MAX_PROCESSES)), multiprocessing.cpu_count()
)

# TODO: Remove these paths bit by bit

# Pre-processed data
DATA_PREPROCESS_DIR = "data/pre-process"

# Results
DATA_RESULT_DIR = "results"
