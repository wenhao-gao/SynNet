"""Computes the fingerprint similarity of molecules in {valid,test}-set to molecules in the training set.
"""  # TODO: clean up, un-nest a couple of fcts
import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from syn_net.utils.data_utils import SyntheticTreeSet

logger = logging.getLogger(__file__)

from syn_net.config import MAX_PROCESSES


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory with `*{train,valid,test}*.json.gz`-data of synthetic trees",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save similarity-values for test,valid-synthetic trees. (*csv.gz)",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def _match_dataset_filename(
    path: str, dataset_type: str
) -> Path:  # TODO: consolidate with code in script/05-*
    """Helper to find the exact filename for {train,valid,test} file."""
    files = list(Path(path).glob(f"*{dataset_type}*.json.gz"))
    if len(files) != 1:
        raise ValueError(f"Can not find unique '{dataset_type} 'file, got {files}")
    return files[0]


def find_similar_fp(fp: np.ndarray, fps_reference: np.ndarray):
    """Finds most similar fingerprint in a reference set for `fp`.
    Uses Tanimoto Similarity.
    """
    dists = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps_reference))
    similarity_score, idx = dists.max(), dists.argmax()
    return similarity_score, idx


def _compute_fp_bitvector(smiles: list[str], radius: int = 2, nbits: int = 1024):
    return [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius, nBits=nbits)
        for smi in smiles
    ]


def get_smiles_and_fps(dataset: str) -> Tuple[list[str], list[np.ndarray]]:
    file = _match_dataset_filename(args.input_dir, dataset)
    syntree_collection = SyntheticTreeSet().load(file)
    smiles = [st.root.smiles for st in syntree_collection]
    fps = _compute_fp_bitvector(smiles)
    return smiles, fps


def compute_most_similar_smiles(
    split: str,
    fps: np.ndarray,
    smiles: list[str],
    /,
    fps_reference: np.ndarray,
    smiles_reference: list[str],
) -> pd.DataFrame:

    func = partial(find_similar_fp, fps_reference=fps_reference)
    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(func, fps)

    similarities, idx = zip(*results)
    most_similiar_ref_smiles = np.asarray(smiles_reference)[np.asarray(idx, dtype=int)]
    # ^ Use numpy for slicing...

    df = pd.DataFrame(
        {
            "split": split,
            "smiles": smiles,
            "most_similar_smiles": most_similiar_ref_smiles,
            "similarity": similarities,
        }
    )
    return df


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")
    # Load data
    smiles_train, fps_train = get_smiles_and_fps("train")
    smiles_valid, fps_valid = get_smiles_and_fps("valid")
    smiles_test, fps_test = get_smiles_and_fps("test")

    # Compute (mp)
    logger.info("Start computing most similar smiles...")
    df_valid = compute_most_similar_smiles(
        "valid", fps_valid, smiles_valid, fps_reference=fps_train, smiles_reference=smiles_train
    )
    df_test = compute_most_similar_smiles(
        "test", fps_test, smiles_test, fps_reference=fps_train, smiles_reference=smiles_train
    )
    logger.info("Computed most similar smiles for {valid,test}-set.")

    # Save
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([df_valid, df_test], axis=0, ignore_index=True)
    df.to_csv(args.output_file, index=False, compression="gzip")
    logger.info(f"Successfully saved output to {args.output_file}.")

    logger.info("Completed.")
