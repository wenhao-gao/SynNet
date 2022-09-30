"""Central place for all configuration, paths, and parameter."""
import multiprocessing

# Multiprocessing
MAX_PROCESSES = min(32, multiprocessing.cpu_count()) - 1

# TODO: Remove these paths bit by bit (not used except for decoing as of now)
# Paths
DATA_DIR = "data"
ASSETS_DIR = "data/assets"

#
BUILDING_BLOCKS_RAW_DIR = f"{ASSETS_DIR}/building-blocks"
REACTION_TEMPLATE_DIR = f"{ASSETS_DIR}/reaction-templates"

# Pre-processed data
DATA_PREPROCESS_DIR = "data/pre-process"
DATA_EMBEDDINGS_DIR = "data/pre-process/embeddings"

# Prepared data
DATA_FEATURIZED_DIR = "data/featurized"

# Results
DATA_RESULT_DIR = "results"

# Checkpoints (& pre-trained weights)
CHECKPOINTS_DIR = "checkpoints"
