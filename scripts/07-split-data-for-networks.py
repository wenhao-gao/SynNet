"""Split the featurized data into X,y-chunks for the {act,rt1,rxn,rt2}-networks
"""
import logging
from pathlib import Path
import json

from syn_net.utils.prep_utils import split_data_into_Xy

logger = logging.getLogger(__file__)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory for the featurized synthetic trees (with {train,valid,test}-data).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Split datasets for each MLP
    logger.info("Start splitting data.")
    num_rxn = 91 # Auxiliary var for indexing TODO: Dont hardcode
    out_dim = 256 # Auxiliary var for indexing TODO: Dont hardcode
    input_dir = Path(args.input_dir)
    output_dir = input_dir / "Xy"
    for dataset_type in "train valid test".split():
        logger.info(f"Split {dataset_type}-data...")
        split_data_into_Xy(
            dataset_type=dataset_type,
            steps_file=input_dir / f"{dataset_type}_steps.npz",
            states_file=input_dir / f"{dataset_type}_states.npz",
            output_dir=input_dir / "Xy",
            num_rxn=num_rxn,
            out_dim=out_dim,
            )

    logger.info(f"Completed.")
