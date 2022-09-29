"""Filter out building blocks that cannot react with any template.
"""
import logging

from rdkit import RDLogger

from syn_net.config import MAX_PROCESSES
from syn_net.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilter,
    ReactionTemplateFileHandler,
)
from syn_net.utils.data_utils import ReactionSet

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)
import json


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        help="File with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxn-templates-file",
        type=str,
        help="Input file with reaction templates as SMARTS(No header, one per line).",
    )
    parser.add_argument(
        "--output-bblock-file",
        type=str,
        help="Output file for the filtered building-blocks.",
    )
    parser.add_argument(
        "--output-rxns-file",
        type=str,
        help="Output file for the collection of reactions matched with building-blocks.",
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

    # Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    rxn_templates = ReactionTemplateFileHandler().load(args.rxn_templates_file)

    bbf = BuildingBlockFilter(
        building_blocks=bblocks,
        rxn_templates=rxn_templates,
        verbose=args.verbose,
        processes=args.ncpu,
    )
    # Time intensive task...
    bbf.filter()

    # ... and save to disk
    bblocks_filtered = bbf.building_blocks_filtered
    BuildingBlockFileHandler().save(args.output_bblock_file, bblocks_filtered)

    # Save collection of reactions which have "available reactants" set (for convenience)
    rxn_collection = ReactionSet(bbf.rxns)
    rxn_collection.save(args.output_rxns_file)

    logger.info(f"Total number of building blocks {len(bblocks):d}")
    logger.info(f"Matched number of building blocks {len(bblocks_filtered):d}")
    logger.info(
        f"{len(bblocks_filtered)/len(bblocks):.2%} of building blocks applicable for the reaction template."
    )

    logger.info("Completed.")
