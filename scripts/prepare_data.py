"""
Prepares the training, testing, and validation data by reading in the states
and steps for the reaction data and re-writing it as separate one-hot encoded
Action, Reactant 1, Reactant 2, and Reaction files.
"""
import logging
from pathlib import Path

from syn_net.config import DATA_FEATURIZED_DIR
from syn_net.utils.prep_utils import prep_data

logger = logging.getLogger(__file__)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(DATA_FEATURIZED_DIR)) + "/hb_fp_2_4096_fp_256", # TODO: dont hardcode
        help="Input directory for the featurized synthetic trees (with {train,valid,test}-data).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    logger.info("Start.")
    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {vars(args)}")

    featurized_data_dir = args.input_dir

    # Split datasets for each MLP
    logger.info("Start splitting data.")
    num_rxn = 91 # Auxiliary var for indexing TODO: Dont hardcode
    out_dim = 256 # Auxiliary var for indexing TODO: Dont hardcode
    prep_data(featurized_data_dir, num_rxn, out_dim)

    logger.info(f"Completed.")
