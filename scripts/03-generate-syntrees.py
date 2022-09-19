import logging
from collections import Counter
from pathlib import Path

from rdkit import RDLogger

from syn_net.config import DATA_PREPROCESS_DIR, MAX_PROCESSES
from syn_net.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    ReactionTemplateFileHandler,
)
from syn_net.data_generation.syntrees import SynTreeGenerator, wraps_syntreegenerator_generate
from syn_net.utils.data_utils import SyntheticTree, SyntheticTreeSet

logger = logging.getLogger(__name__)
from typing import Union

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
        default=Path(DATA_PREPROCESS_DIR) / f"synthetic-trees.json.gz",
        help="Output file for the generated synthetic trees (*.json.gz)",
    )
    # Parameters
    parser.add_argument("--number-syntrees", type=int, help="Number of SynTrees to generate.")

    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


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
    outcomes: dict[int, str] = dict()
    syntrees: list[Union[SyntheticTree, None]] = []
    for i in range(args.number_syntrees):
        st, e = wraps_syntreegenerator_generate(stgen)
        outcomes[i] = e.__class__.__name__ if e is not None else "success"
        syntrees.append(st)
    logger.info(f"SynTree generation completed. Results: {Counter(outcomes.values())}")

    # Save synthetic trees on disk
    syntree_collection = SyntheticTreeSet(syntrees)
    syntree_collection.save(args.output_file)

    logger.info(f"Completed.")
