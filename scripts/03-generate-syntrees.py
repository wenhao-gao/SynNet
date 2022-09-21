import logging
from collections import Counter

from rdkit import RDLogger
from tqdm import tqdm

from syn_net.config import MAX_PROCESSES
from syn_net.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    ReactionTemplateFileHandler,
)
from syn_net.data_generation.syntrees import SynTreeGenerator, wraps_syntreegenerator_generate
from syn_net.utils.data_utils import SyntheticTree, SyntheticTreeSet

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
        default="data/pre-process/building-blocks/enamine-us-smiles.csv.gz",  # TODO: change
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        default="data/assets/reaction-templates/hb.txt",  # TODO: change
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
        "--number-syntrees", type=int, default=1000, help="Number of SynTrees to generate."
    )

    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def generate_mp() -> Tuple[dict[int, str], list[Union[SyntheticTree, None]]]:
    from functools import partial

    import numpy as np
    from pathos import multiprocessing as mp

    def wrapper(stgen, _):
        stgen.rng = np.random.default_rng()
        return wraps_syntreegenerator_generate(stgen)

    func = partial(wrapper, stgen)
    with mp.Pool(processes=4) as pool:
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
    logger.info(f"Arguments: {vars(args)}")

    # Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)

    # Init SynTree Generator
    stgen = SynTreeGenerator(
        building_blocks=bblocks, rxn_templates=rxn_templates, verbose=args.verbose
    )
    # Generate synthetic trees
    logger.info(f"Start generation of {args.number_syntrees} SynTrees...")
    if args.ncpu > 1:
        outcomes, syntrees = generate_mp()
    else:
        outcomes, syntrees = generate()
    logger.info(f"SynTree generation completed. Results: {Counter(outcomes.values())}")

    # Save synthetic trees on disk
    syntree_collection = SyntheticTreeSet(syntrees)
    syntree_collection.save(args.output_file)

    logger.info(f"Completed.")
