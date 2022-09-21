"""
Reads synthetic tree data and splits it into training, validation and testing sets.
"""
from syn_net.utils.data_utils import SyntheticTreeSet
from pathlib import Path
from syn_net.config import DATA_PREPROCESS_DIR, DATA_PREPARED_DIR, MAX_PROCESSES
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
    st_set = SyntheticTreeSet()
    file =  args.input_file
    print(f'Reading data from {file}')
    st_set.load(file)
    data = st_set.sts
    del st_set
    num_total = len(data)
    print(f"There are {len(data)} synthetic trees.")

    # Split data
    SPLIT_RATIO = [0.6, 0.2, 0.2]

    num_train = int(SPLIT_RATIO[0] * num_total)
    num_valid = int(SPLIT_RATIO[1] * num_total)
    num_test = num_total - num_train - num_valid

    data_train = data[:num_train]
    data_valid = data[num_train: num_train + num_valid]
    data_test = data[num_train + num_valid: ]

    # Save to local disk
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    print("Saving training dataset: ", len(data_train))
    trees = SyntheticTreeSet(data_train)
    trees.save(out_dir / "synthetic-trees-train.json.gz")

    print("Saving validation dataset: ", len(data_valid))
    trees = SyntheticTreeSet(data_valid)
    trees.save(out_dir / "synthetic-trees-valid.json.gz")

    print("Saving testing dataset: ", len(data_test))
    trees = SyntheticTreeSet(data_test)
    trees.save(out_dir / "synthetic-trees-test.json.gz")

    print("Finish!")
