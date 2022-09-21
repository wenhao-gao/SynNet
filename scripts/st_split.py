"""
Reads synthetic tree data and splits it into training, validation and testing sets.
"""
from syn_net.utils.data_utils import SyntheticTreeSet
from pathlib import Path
from syn_net.config import DATA_PREPROCESS_DIR, MAX_PROCESSES
import json
import logging
logger = logging.getLogger(__name__)

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/pre-process/synthetic-trees.json.gz",
        help="Input file for the filtered generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(DATA_PREPROCESS_DIR) / "split"),
        help="Output directory for the splitted synthetic trees (*.json.gz)",
    )

    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load filtered synthetic trees
    logger.info(f'Reading data from {args.input_file}')
    syntree_collection = SyntheticTreeSet().load(args.input_file)
    syntrees = syntree_collection.sts

    num_total = len(syntrees)
    logger.info(f"There are {len(syntrees)} synthetic trees.")

    # Split data
    SPLIT_RATIO = [0.6, 0.2, 0.2]

    num_train = int(SPLIT_RATIO[0] * num_total)
    num_valid = int(SPLIT_RATIO[1] * num_total)
    num_test = num_total - num_train - num_valid

    data_train = syntrees[:num_train]
    data_valid = syntrees[num_train: num_train + num_valid]
    data_test = syntrees[num_train + num_valid: ]

    # Save to local disk
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    logger.info(f"Saving training dataset. Number of syntrees: {len(data_train)}")
    SyntheticTreeSet(data_train).save(out_dir / "synthetic-trees-train.json.gz")

    logger.info(f"Saving validation dataset. Number of syntrees: {len(data_valid)}")
    SyntheticTreeSet(data_valid).save(out_dir / "synthetic-trees-valid.json.gz")

    logger.info(f"Saving testing dataset. Number of syntrees: {len(data_test)}")
    SyntheticTreeSet(data_test).save(out_dir / "synthetic-trees-test.json.gz")

    logger.info(f"Completed.")

