"""Filter Synthetic Trees.
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Union

import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm

from synnet.config import MAX_PROCESSES
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet

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
        help="Input file for the filtered generated synthetic trees (*.json.gz)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for the filtered generated synthetic trees (*.json.gz)",
    )

    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def filter_syntree(syntree: SyntheticTree) -> Union[SyntheticTree, int]:
    """Apply filters to `syntree` and return it, if all filters are passed. Else, return error code."""
    # Filter 1: Is root molecule valid?
    keep_tree = valid_root_mol_filter.filter(syntree)
    if not keep_tree:
        return -1

    # Filter 2: Is root molecule "pharmaceutically interesting?"
    keep_tree = interesting_mol_filter.filter(syntree)
    if not keep_tree:
        return -2

    # We passed all filters. This tree ascended to our dataset
    return syntree


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load previously generated synthetic trees
    syntree_collection = SyntheticTreeSet().load(args.input_file)
    logger.info(f"Successfully loaded '{args.input_file}' with {len(syntree_collection)} syntrees.")

    # Filter trees
    # TODO: Move to src/synnet/data_generation/filters.py ?
    valid_root_mol_filter = ValidRootMolFilter()
    interesting_mol_filter = OracleFilter(threshold=0.5, rng=np.random.default_rng(42))

    syntrees = [s for s in syntree_collection if s is not None]

    logger.info(f"Start filtering {len(syntrees)} syntrees.")

    if args.ncpu == 1:
        syntrees = tqdm(syntrees) if args.verbose else syntrees
        results = [filter_syntree(syntree) for syntree in syntree_collection]
    else:
        with mp.Pool(processes=args.ncpu) as pool:
            logger.info(f"Starting MP with ncpu={args.ncpu}")
            results = pool.map(filter_syntree, syntrees)

    logger.info("Finished decoding.")

    # Handle results, most notably keep track of why we deleted the tree
    outcomes: dict[str, int] = {
        "invalid_root_mol": 0,
        "not_interesting": 0,
    }
    syntrees_filtered = []
    for res in results:
        if res == -1:
            outcomes["invalid_root_mol"] += 1
        if res == -2:
            outcomes["not_interesting"] += 1
        else:
            syntrees_filtered.append(res)

    logger.info(f"Successfully filtered syntrees.")

    # Save filtered synthetic trees on disk
    SyntheticTreeSet(syntrees_filtered).save(args.output_file)
    logger.info(f"Successfully saved '{args.output_file}' with {len(syntrees_filtered)} syntrees.")

    summary_file = Path(args.output_file).parent / "filter-summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(outcomes, indent=2))

    logger.info(f"Completed.")
