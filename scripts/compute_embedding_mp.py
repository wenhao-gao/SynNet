"""
Computes the molecular embeddings of the purchasable building blocks.

The embeddings are also referred to as "output embedding". 
In the embedding space, a kNN-search will identify the 1st or 2nd reactant.
"""
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from syn_net.config import DATA_EMBEDDINGS_DIR, DATA_PREPROCESS_DIR
from syn_net.utils.predict_utils import fp_256, fp_512, fp_1024, fp_2048, fp_4096, mol_embedding, rdkit2d_embedding

logger = logging.getLogger(__file__)


FUNCTIONS = {
    "gin": mol_embedding,
    "fp_4096": fp_4096,
    "fp_2048": fp_2048,
    "fp_1024": fp_1024,
    "fp_512": fp_512,
    "fp_256": fp_256,
    "rdkit2d": rdkit2d_embedding,
}


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="fp_256", choices=FUNCTIONS.keys(), help="Objective function to optimize")
    parser.add_argument("--ncpu", type=int, default=64, help="Number of cpus")
    parser.add_argument("-rxn", "--rxn_template", type=str, default="hb", choices=["hb", "pis"], help="Choose from ['hb', 'pis']")
    parser.add_argument("--input", type=str, help="Input file with SMILES strings (One per line).")
    args = parser.parse_args()

    reaction_template_id = args.rxn_template
    building_blocks_id = "enamine_us-2021-smiles"

    # Load building blocks
    file = Path(DATA_PREPROCESS_DIR) / f"{reaction_template_id}-{building_blocks_id}-matched.csv.gz"

    data = pd.read_csv(file)["SMILES"].tolist()
    logger.info(f"Successfully read {file}.")
    logger.info(f"Total number of building blocks: {len(data)}.")

    func = FUNCTIONS[args.feature]
    with mp.Pool(processes=args.ncpu) as pool:
        embeddings = pool.map(func, data)

    # Save embeddings
    embeddings = np.array(embeddings)

    path = Path(DATA_EMBEDDINGS_DIR)
    path.mkdir(exist_ok=1, parents=1)
    outfile = path / f"{reaction_template_id}-{building_blocks_id}-embeddings.npy"

    np.save(outfile, embeddings)
    logger.info(f"Successfully saved to {outfile}.")
