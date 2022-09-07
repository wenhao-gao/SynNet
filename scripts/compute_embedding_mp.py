"""
Computes the molecular embeddings of the purchasable building blocks.

The embeddings are also referred to as "output embedding".
In the embedding space, a kNN-search will identify the 1st or 2nd reactant.
"""
import logging
from pathlib import Path

import pandas as pd

from syn_net.MolEmbedder import MolEmbedder
from syn_net.config import DATA_EMBEDDINGS_DIR, DATA_PREPROCESS_DIR
from syn_net.encoding.fingerprints import fp_256, fp_512, fp_1024, fp_2048, fp_4096
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

def _load_building_blocks(file: Path) -> list[str]:
    return pd.read_csv(file)["SMILES"].to_list()

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-blocks-file", type=str, help="Input file with SMILES strings (First row `SMILES`, then one per line).")
    parser.add_argument("--output-file", type=str, help="Output file for the computed embeddings.")
    parser.add_argument("--feature", type=str, default="fp_256", choices=FUNCTIONS.keys(), help="Objective function to optimize")
    parser.add_argument("--ncpu", type=int, default=32, help="Number of cpus")
    # Command line args to be deprecated, only support input/output file in future.
    parser.add_argument("--rxn-template", type=str, default="hb", choices=["hb", "pis"], help="Choose from ['hb', 'pis']")
    parser.add_argument("--building-blocks-id", type=str, default="enamine_us-2021-smiles")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    # Load building blocks
    if (file := args.building_blocks_file) is None:
        # Try to construct filename
        file = Path(DATA_PREPROCESS_DIR) / f"{args.rxn_template}-{args.building_blocks_id}-matched.csv.gz"
    bblocks = _load_building_blocks(file)
    logger.info(f"Successfully read {file}.")
    logger.info(f"Total number of building blocks: {len(bblocks)}.")

    # Compute embeddings
    func = FUNCTIONS[args.feature]
    molembedder = MolEmbedder(processes=args.ncpu).compute_embeddings(func,bblocks)

    # Save?
    if (outfile := args.output_file) is None:
        # Try to construct filename
        outfile = Path(DATA_EMBEDDINGS_DIR) / f"{args.rxn_template}-{args.building_blocks_id}-{args.feature}.npy"
    molembedder.save_precomputed(outfile)

