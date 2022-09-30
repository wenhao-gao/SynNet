"""Filter Synthetic Trees.
"""

import json
import logging

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm

from syn_net.config import MAX_PROCESSES
from syn_net.utils.data_utils import SyntheticTree, SyntheticTreeSet

logger = logging.getLogger(__name__)

RDLogger.DisableLog("rdApp.*")


class Filter:
    def filter(self, st: SyntheticTree, **kwargs) -> bool:
        ...


class ValidRootMolFilter(Filter):
    def filter(self, st: SyntheticTree, **kwargs) -> bool:
        return Chem.MolFromSmiles(st.root.smiles) is not None


class OracleFilter(Filter):
    def __init__(
        self,
        name: str = "qed",
        threshold: float = 0.5,
        rng=np.random.default_rng(42),
    ) -> None:
        super().__init__()
        from tdc import Oracle

        self.oracle_fct = Oracle(name=name)
        self.threshold = threshold
        self.rng = rng

    def _qed(self, st: SyntheticTree):
        """Filter for molecules with a high qed."""
        return self.oracle_fct(st.root.smiles) > self.threshold

    def _random(self, st: SyntheticTree):
        """Filter molecules that fail the `_qed` filter; i.e. randomly select low qed molecules."""
        return self.rng.random() < (self.oracle_fct(st.root.smiles) / self.threshold)

    def filter(self, st: SyntheticTree) -> bool:
        return self._qed(st) or self._random(st)


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
        "--output-file",
        type=str,
        default="data/pre-process/synthetic-trees-filtered.json.gz",
        help="Output file for the filtered generated synthetic trees (*.json.gz)",
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

    # Load previously generated synthetic trees
    syntree_collection = SyntheticTreeSet().load(args.input_file)
    logger.info(f"Successfully loaded '{args.input_file}' with {len(syntree_collection)} syntrees.")

    # Filter trees
    # TODO: Move to src/syn_net/data_generation/filters.py ?
    valid_root_mol_filter = ValidRootMolFilter()
    interesting_mol_filter = OracleFilter(threshold=0.5, rng=np.random.default_rng())

    logger.info(f"Start filtering syntrees...")
    syntrees = []
    syntree_collection = [s for s in syntree_collection if s is not None]
    syntree_collection = tqdm(syntree_collection) if args.verbose else syntree_collection
    outcomes: dict[int, str] = dict()  # TODO: think about what metrics to track here
    for i, st in enumerate(syntree_collection):

        # Filter 1: Is root molecule valid?
        keep_tree = valid_root_mol_filter.filter(st)
        if not keep_tree:
            continue

        # Filter 2: Is root molecule "pharmaceutically interesting?"
        keep_tree = interesting_mol_filter.filter(st)
        if not keep_tree:
            continue

        # We passed all filters. This tree ascended to our dataset
        syntrees.append(st)
    logger.info(f"Successfully filtered syntrees.")

    # Save filtered synthetic trees on disk
    SyntheticTreeSet(syntrees).save(args.output_file)
    logger.info(f"Successfully saved '{args.output_file}' with {len(syntrees)} syntrees.")

    logger.info(f"Completed.")
