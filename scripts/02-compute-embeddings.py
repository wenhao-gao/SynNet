"""
Computes the molecular embeddings of the purchasable building blocks.

The embeddings are also referred to as "output embedding".
In the embedding space, a kNN-search will identify the 1st or 2nd reactant.
"""

import json
import logging
from functools import partial

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.fingerprints import mol_fp
from synnet.MolEmbedder import MolEmbedder

logger = logging.getLogger(__file__)

FUNCTIONS = {
    "fp_4096": partial(mol_fp, _radius=2, _nBits=4096),
    "fp_2048": partial(mol_fp, _radius=2, _nBits=2048),
    "fp_1024": partial(mol_fp, _radius=2, _nBits=1024),
    "fp_512": partial(mol_fp, _radius=2, _nBits=512),
    "fp_256": partial(mol_fp, _radius=2, _nBits=256),
}  # TODO: think about refactor/merge with `MorganFingerprintEncoder`


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
        "--output-file",
        type=str,
        help="Output file for the computed embeddings file. (*.npy)",
    )
    parser.add_argument(
        "--featurization-fct",
        type=str,
        choices=FUNCTIONS.keys(),
        help="Featurization function applied to each molecule.",
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

    # Load building blocks
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    logger.info(f"Successfully read {args.building_blocks_file}.")
    logger.info(f"Total number of building blocks: {len(bblocks)}.")

    # Compute embeddings
    func = FUNCTIONS[args.featurization_fct]
    molembedder = MolEmbedder(processes=args.ncpu).compute_embeddings(func, bblocks)

    # Save?
    molembedder.save_precomputed(args.output_file)

    logger.info("Completed.")
