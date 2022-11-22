"""Generate synthetic trees.
"""  # TODO: clean up this mess
import json
import logging
from collections import Counter
from pathlib import Path

from rdkit import RDLogger
from tqdm import tqdm

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    ReactionTemplateFileHandler,
)
from synnet.data_generation.syntrees import SynTreeGenerator, wraps_syntreegenerator_generate
from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet

logger = logging.getLogger(__name__)
from typing import Tuple, Union

RDLogger.DisableLog("rdApp.*")

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        help="Input file with reaction templates as SMARTS(No header, one per line).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/pre-precess/synthetic-trees.json.gz",
        help="Output file for the generated synthetic trees (*.json.gz)",
    )
    # Parameters
    parser.add_argument(
        "--number-syntrees", type=int, default=100, help="Number of SynTrees to generate."
    )

    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


def generate_mp() -> Tuple[dict[int, str], list[Union[SyntheticTree, None]]]:
    from functools import partial

    import numpy as np
    from pathos import multiprocessing as mp

    def wrapper(stgen, _):
        stgen.rng = np.random.default_rng()  # TODO: Think about this...
        return wraps_syntreegenerator_generate(stgen)

    func = partial(wrapper, stgen)

    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(func, range(args.number_syntrees))

    outcomes = {
        i: e.__class__.__name__ if e is not None else "success" for i, (_, e) in enumerate(results)
    }
    syntrees = [st for (st, e) in results if e is None]
    return outcomes, syntrees


def generate() -> Tuple[dict[int, str], list[Union[SyntheticTree, None]]]:
    outcomes: dict[int, str] = dict()
    syntrees: list[Union[SyntheticTree, None]] = []
    myrange = tqdm(range(args.number_syntrees)) if args.verbose else range(args.number_syntrees)
    for i in myrange:
        st, e = wraps_syntreegenerator_generate(stgen)
        outcomes[i] = e.__class__.__name__ if e is not None else "success"
        syntrees.append(st)

    return outcomes, syntrees


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    if args.debug:
        st_logger = logging.getLogger("synnet.data_generation.syntrees")
        st_logger.setLevel("DEBUG")
        RDLogger.EnableLog("rdApp.*")

    # Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)
    logger.info("Loaded building block & rxn-template assets.")

    # Init SynTree Generator
    logger.info("Start initializing SynTreeGenerator...")
    stgen = SynTreeGenerator(
        building_blocks=bblocks, rxn_templates=rxn_templates, verbose=args.verbose
    )
    logger.info("Successfully initialized SynTreeGenerator.")

    # Generate synthetic trees
    logger.info(f"Start generation of {args.number_syntrees} SynTrees...")
    if args.ncpu > 1:
        outcomes, syntrees = generate_mp()
    else:
        outcomes, syntrees = generate()
    result_summary = Counter(outcomes.values())
    logger.info(f"SynTree generation completed. Results: {result_summary}")

    summary_file = Path(args.output_file).parent / "results-summary.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(result_summary, indent=2))

    # Save synthetic trees on disk
    syntree_collection = SyntheticTreeSet(syntrees)
    syntree_collection = SyntheticTreeSet(
        [st for st in syntree_collection if st is not None and st.depth>1]
    )
    syntree_collection.save(args.output_file)

    logger.info(f"Generated syntrees: {len(syntree_collection)}")
    logger.info(f"Completed.")
