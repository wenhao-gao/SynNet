import logging
import os

RUNNING_ON_HPC: bool = "SLURM_JOB_ID" in os.environ

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
    # handlers=[logging.FileHandler(".log"),logging.StreamHandler()],
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
