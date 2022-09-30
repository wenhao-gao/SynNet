"""Central place for all configuration, paths, and parameter."""
import multiprocessing

# Multiprocessing
MAX_PROCESSES = min(32, multiprocessing.cpu_count()) - 1

# TODO: Remove these paths bit by bit

# Pre-processed data
DATA_PREPROCESS_DIR = "data/pre-process"

# Prepared data
DATA_FEATURIZED_DIR = "data/featurized"

# Results
DATA_RESULT_DIR = "results"

# Checkpoints (& pre-trained weights)
CHECKPOINTS_DIR = "checkpoints"
