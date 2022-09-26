"""
Splits a synthetic tree into states and steps.
"""
import json
import logging
from pathlib import Path

from scipy import sparse
from tqdm import tqdm

from syn_net.data_generation.syntrees import SynTreeFeaturizer
from syn_net.utils.data_utils import SyntheticTreeSet

logger = logging.getLogger(__file__)

from syn_net.config import DATA_FEATURIZED_DIR


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/pre-process/split/synthetic-trees-valid.json.gz",  # TODO think about filename vs dir
        help="Input file for the splitted generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(DATA_FEATURIZED_DIR)) + "debug-newversion",
        help="Output directory for the splitted synthetic trees (*.json.gz)",
    )
    return parser.parse_args()


def _extract_dataset(filename: str) -> str:
    stem = Path(filename).stem.split(".")[0]
    return stem.split("-")[-1]  # TODO: avoid hard coding


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")
    dataset_type = _extract_dataset(args.input_file)

    st_set = SyntheticTreeSet().load(args.input_file)
    logger.info(f"Number of synthetic trees: {len(st_set.sts)}")
    data: list = st_set.sts
    del st_set

    # Start splitting synthetic trees in states and steps
    states = []
    steps = []
    stf = SynTreeFeaturizer()
    for st in tqdm(data):
        try:
            state, step = stf.featurize(st)
        except Exception as e:
            logger.exception(e, exc_info=e)
            continue
        states.append(state)
        steps.append(step)

    # Set output directory
    save_dir = Path(args.output_dir) / "hb_fp_2_4096_fp_256"  # TODO: Save info as json in dir?
    Path(save_dir).mkdir(parents=1, exist_ok=1)
    dataset_type = _extract_dataset(args.input_file)

    # Finally, save.
    logger.info(f"Saving to {save_dir}")
    states = sparse.vstack(states)
    steps = sparse.vstack(steps)
    sparse.save_npz(save_dir / f"states_{dataset_type}.npz", states)
    sparse.save_npz(save_dir / f"steps_{dataset_type}.npz", steps)

    logger.info("Save successful.")
    logger.info("Completed.")
