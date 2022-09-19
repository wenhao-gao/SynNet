"""
Computes the molecular embeddings of the purchasable building blocks.

The embeddings are also referred to as "output embedding".
In the embedding space, a kNN-search will identify the 1st or 2nd reactant.
"""

import logging

from syn_net.data_generation.preprocessing import BuildingBlockFileHandler
from syn_net.encoding.fingerprints import fp_256, fp_512, fp_1024, fp_2048, fp_4096
from syn_net.MolEmbedder import MolEmbedder
from syn_net.config import MAX_PROCESSES
# from syn_net.encoding.gins import mol_embedding
# from syn_net.utils.prep_utils import rdkit2d_embedding


logger = logging.getLogger(__file__)


FUNCTIONS = {
    # "gin": mol_embedding,
    "fp_4096": fp_4096,
    "fp_2048": fp_2048,
    "fp_1024": fp_1024,
    "fp_512": fp_512,
    "fp_256": fp_256,
    # "rdkit2d": rdkit2d_embedding,
}


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
        help="Output file for the computed embeddings file. (*.npy)",
    )
    parser.add_argument(
        "--featurization-fct",
        type=str,
        default="fp_256",
        choices=FUNCTIONS.keys(),
        help="Objective function to optimize",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # Load building blocks
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    logger.info(f"Successfully read {args.building_blocks_file}.")
    logger.info(f"Total number of building blocks: {len(bblocks)}.")

    # Compute embeddings
    func = FUNCTIONS[args.featurization_fct]
    molembedder = MolEmbedder(processes=args.ncpu).compute_embeddings(func, bblocks)

    # Save?
    molembedder.save_precomputed(args.output_file)
