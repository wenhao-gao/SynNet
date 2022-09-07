"""Filter out building blocks that cannot react with any template.
"""
import logging

from rdkit import RDLogger
from syn_net.data_generation.preprocessing import BuildingBlockFileHandler, BuildingBlockFilter

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


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
        help="Output file for the filtered building-blocks file.",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=32, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logger.info("Start.")

    # Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    with open(args.rxn_templates_file, "rt") as f:
        rxn_templates = f.readlines()

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
    BuildingBlockFileHandler().save(args.output_file, bblocks_filtered)

    logger.info(f"Total number of building blocks {len(bblocks):d}")
    logger.info(f"Matched number of building blocks {len(bblocks_filtered):d}")
    logger.info(
        f"{len(bblocks_filtered)/len(bblocks):.2%} of building blocks applicable for the reaction template."
    )

    logger.info("Completed.")
