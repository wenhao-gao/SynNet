"""
This file contains functions for generating molecular embeddings from SMILES using GIN.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from syn_net.utils.predict_utils import mol_embedding, fp_embedding, rdkit2d_embedding


def get_mol_embedding_func(feature):
    """
    Returns the molecular embedding function.

    Args:
        feature (str): Indicates the type of featurization to use (GIN or Morgan
            fingerprint), and the size.

    Returns:
        Callable: The embedding function.
    """
    if feature == 'gin':
        embedding_func = lambda smi: mol_embedding(smi, device='cpu')
    elif feature == 'fp_4096':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=4096)
    elif feature == 'fp_2048':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=2048)
    elif feature == 'fp_1024':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=1024)
    elif feature == 'fp_512':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=512)
    elif feature == 'fp_256':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=256)
    elif feature == 'rdkit2d':
        embedding_func = rdkit2d_embedding
    return embedding_func

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="gin",
                        help="Objective function to optimize")
    parser.add_argument("--ncpu", type=int, default=16,
                        help="Number of cpus")
    args = parser.parse_args()

    path = '/pool001/whgao/data/synth_net/st_hb/'
    ## path = './tests/data/'  ## for debugging
    data = pd.read_csv(path + 'enamine_us_matched.csv.gz', compression='gzip')['SMILES'].tolist()
    ## data = pd.read_csv(path + 'building_blocks_matched.csv.gz', compression='gzip')['SMILES'].tolist()  ## for debugging
    print('Total data: ', len(data))

    embeddings = []
    for smi in tqdm(data):
        embeddings.append(mol_embedding(smi))

    embedding = np.array(embeddings)
    np.save(path + 'enamine_us_emb_' + args.feature + '.npy', embeddings)

    print('Finish!')
