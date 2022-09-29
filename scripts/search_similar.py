"""
Computes the fingerprint similarity of molecules in the validation and test set to
molecules in the training set.
"""
import numpy as np
import pandas as pd
from syn_net.utils.data_utils import *
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing as mp
from rdkit import DataStructs
import logging
from pathlib import Path

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
        help="Optional: File to save similarity-values for test,valid-synthetic trees.",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def _match_dataset_filename(path: str, dataset_type: str) -> Path: # TODO: consolidate with code in script/05-*
    """Helper to find the exact filename for {train,valid,test} file."""
    files = list(Path(path).glob(f"*{dataset_type}*.json.gz"))
    if len(files) != 1:
        raise ValueError(f"Can not find unique '{dataset_type} 'file, got {files}")
    return files[0]

def func(fp: np.ndarray, fps_reference: np.ndarray):
    """Finds most similar fingerprint in a reference set for `fp`.
    Uses Tanimoto Similarity.
    """
    dists = np.array(
        [DataStructs.FingerprintSimilarity(fp, fp_, metric=DataStructs.TanimotoSimilarity) for fp_ in fps_train])
    similarity_score, idx = dists.max(), dists.argmax()
    return similarity_score, idx

def _compute_fp_bitvector(smiles: list[str], radius: int=2, nbits: int=1024):
     return [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius, nBits=nbits) for smi in smiles]

def _save_df(file: str, df):
    if file is None: return
    df.to_csv(file, index=False)

if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    file = '/pool001/whgao/data/synth_net/st_hb/st_train.json.gz'
    syntree_collection = SyntheticTreeSet().load(file)
    data_train = [st.root.smiles for st in syntree_collection]
    fps_train = _compute_fp_bitvector(data_train)

    file = '/pool001/whgao/data/synth_net/st_hb/st_test.json.gz'
    syntree_collection = SyntheticTreeSet().load(file)
    data_test = [st.root.smiles for st in syntree_collection]
    fps_test = _compute_fp_bitvector(data_test)

    file = '/pool001/whgao/data/synth_net/st_hb/st_valid.json.gz'
    syntree_collection = SyntheticTreeSet().load(file)
    data_valid = [st.root.smiles for st in syntree_collection]
    fps_valid = _compute_fp_bitvector(data_valid)

    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(func, fps_valid)

    similaritys = [r[0] for r in results]
    indices = [data_train[r[1]] for r in results]
    df1 = pd.DataFrame({'smiles': data_valid, 'split': 'valid', 'most similar': indices, 'similarity': similaritys})

    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(func, fps_test)

    similaritys = [r[0] for r in results]
    indices = [data_train[r[1]] for r in results]
    df2 = pd.DataFrame({'smiles': data_test, 'split': 'test', 'most similar': indices, 'similarity': similaritys})

    outfile = 'data_similarity.csv'
    _save_df(outfile,  pd.concat([df1, df2], axis=0, ignore_index=True))


    print('Finish!')
