"""Splits a synthetic tree into states and steps.
"""
import json
import logging
from pathlib import Path

from scipy import sparse
from tqdm import tqdm

from syn_net.data_generation.syntrees import (
    IdentityIntEncoder,
    MorganFingerprintEncoder,
    SynTreeFeaturizer,
)
from syn_net.utils.data_utils import SyntheticTreeSet

logger = logging.getLogger(__file__)

from syn_net.config import MAX_PROCESSES


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory with `*{train,valid,test}*.json.gz`-data of synthetic trees",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for the splitted synthetic trees ({train,valid,test}_{steps,states}.npz",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def _match_dataset_filename(path: str, dataset_type: str) -> Path:
    """Helper to find the exact filename for {train,valid,test} file."""
    files = list(Path(path).glob(f"*{dataset_type}*.json.gz"))
    if len(files) != 1:
        raise ValueError(f"Can not find unique '{dataset_type} 'file, got {files}")
    return files[0]


def featurize_data(
    syntree_featurizer: SynTreeFeaturizer, input_dir: str, output_dir: Path, verbose: bool = False
):
    """Wrapper method to featurize synthetic tree data."""

    # Load syntree data
    logger.info(f"Start loading {input_dir}")
    syntree_collection = SyntheticTreeSet().load(input_dir)
    logger.info(f"Successfully loaded synthetic trees.")
    logger.info(f"  Number of trees: {len(syntree_collection.sts)}")

    # Start splitting synthetic trees in states and steps
    states = []
    steps = []
    unsuccessfuls = []
    it = tqdm(syntree_collection) if verbose else syntree_collection
    for i, syntree in enumerate(it):
        try:
            state, step = syntree_featurizer.featurize(syntree)
        except Exception as e:
            logger.exception(e, exc_info=e)
            unsuccessfuls += [i]
            continue
        states.append(state)
        steps.append(step)
    logger.info(f"Completed featurizing syntrees.")
    if len(unsuccessfuls) > 0:
        logger.warning(f"Unsuccessfully attempted to featurize syntrees: {unsuccessfuls}.")

    # Finally, save.
    logger.info(f"Saving to directory {output_dir}")
    states = sparse.vstack(states)
    steps = sparse.vstack(steps)
    sparse.save_npz(output_dir / f"{dataset_type}_states.npz", states)
    sparse.save_npz(output_dir / f"{dataset_type}_steps.npz", steps)
    logger.info("Save successful.")
    return None


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    stfeat = SynTreeFeaturizer(
        reactant_embedder=MorganFingerprintEncoder(2, 256),
        mol_embedder=MorganFingerprintEncoder(2, 4096),
        rxn_embedder=IdentityIntEncoder(),
        action_embedder=IdentityIntEncoder(),
    )

    # Ensure output dir exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=1, exist_ok=1)

    for dataset_type in "train valid test".split():

        input_file = _match_dataset_filename(args.input_dir, dataset_type)
        featurize_data(stfeat, input_file, output_dir=output_dir, verbose=args.verbose)

    # Save information
    (output_dir / "summary.txt").write_text(f"{stfeat}")  # TODO: Parse as proper json?

    logger.info("Completed.")
