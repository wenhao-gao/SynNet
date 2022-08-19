"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
import _mp_predict_multireactant as predict
from syn_net.utils.data_utils import SyntheticTreeSet
from pathlib import Path
from syn_net.config import DATA_PREPARED_DIR, DATA_RESULT_DIR

Path(DATA_RESULT_DIR).mkdir(exist_ok=True)

def _fetch_data_chembl(name: str) -> list[str]:
    raise NotImplementedError
    df = pd.read_csv(f'{DATA_DIR}/chembl_20k.csv')
    smis_query = df.smiles.to_list()
    return smis_query

def _fetch_data(name: str) -> list[str]:
    if args.data in ["train", "valid", "test"]:
        file = Path(DATA_PREPARED_DIR) / f"synthetic-trees-{args.data}.json.gz"
        print(f'Reading data from {file}')
        sts = SyntheticTreeSet()
        sts.load(file)
        smis_query = [st.root.smiles for st in sts.sts]
    else:
        smis_query = _fetch_data_chembl(name)
    return smis_query

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("--ncpu", type=int, default=16,
                        help="Number of cpus")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="Number of molecules to predict.")
    parser.add_argument("-d", "--data", type=str, default='test',
                        help="Choose from ['train', 'valid', 'test', 'chembl']")
    args = parser.parse_args()

    # load the query molecules (i.e. molecules to decode)
    smiles_queries = _fetch_data(args.data)

    # Select only n queries
    if args.num > 0:
        smiles_queries = smiles_queries[:args.num]

    print(f'Start to decode {len(smiles_queries)} target molecules.')
    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(predict.func, smiles_queries)

    smis_decoded = [r[0] for r in results]
    similarities  = [r[1] for r in results]
    trees        = [r[2] for r in results]

    print('Finish decoding')
    print(f'Recovery rate {args.data}: {np.sum(np.array(similarities) == 1.0) / len(similarities)}')
    print(f'Average similarity {args.data}: {np.mean(np.array(similarities))}')

    print('Saving ......')
    save_path = DATA_RESULT_DIR
    df = pd.DataFrame({'query SMILES' : smiles_queries, 
                       'decode SMILES': smis_decoded, 
                       'similarity'   : similarities})
    df.to_csv(f'{save_path}/decode_result_{args.data}.csv.gz', 
              compression='gzip', 
              index=False)
    
    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(f'{save_path}/decoded_st_{args.data}.json.gz')

    print('Finish!')
