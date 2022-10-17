"""Split the featurized data into X,y-chunks for the {act,rt1,rxn,rt2}-networks
"""
import json
import logging
from pathlib import Path

from scipy import sparse

from synnet.utils.prep_utils import split_data_into_Xy

logger = logging.getLogger(__file__)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory for the featurized synthetic trees (with {train,valid,test}-data).",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=2, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def print_datashapes():
    for k, v in data.items():
        logger.info(f"{k}")
        for k, v in v.items():
            logger.info(f"  {k}: {v.shape}")


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Split datasets for each MLP
    logger.info("Start splitting data.")
    num_rxn = 91  # Auxiliary var for indexing TODO: Dont hardcode
    d_knn_emb = 256  # Auxiliary var for indexing TODO: Dont hardcode

    input_dir = Path(args.input_dir)
    output_dir = input_dir / "Xy"
    output_dir.mkdir(parents=True, exist_ok=True)

    for datasplit in "train valid test".split():
        logger.info(f"Split {datasplit}-data...")

        # Load
        states = sparse.load_npz(input_dir / f"{datasplit}_states.npz")  # (n,3*4096)
        steps = sparse.load_npz(input_dir / f"{datasplit}_steps.npz")  # (n,1+256+1+256+4096)

        # Split
        data = split_data_into_Xy(
            steps=steps,
            states=states,
            num_rxn=num_rxn,
            d_knn_emb=d_knn_emb,
        )
        if args.verbose:
            print_datashapes()

        # Save
        for MODEL_ID in "act rt1 rxn rt2".split() + ["rt1_augmented"]:
            X = data[MODEL_ID]["X"]
            y = data[MODEL_ID]["y"]
            sparse.save_npz(output_dir / f"X_{MODEL_ID}_{datasplit}.npz", X)
            sparse.save_npz(output_dir / f"y_{MODEL_ID}_{datasplit}.npz", y)

    logger.info(f"Completed.")
