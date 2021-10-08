import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import json
import _mp_predict_multireactant as predict
from synth_net.utils.data_utils import Reaction, ReactionSet, SyntheticTree, SyntheticTreeSet


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

    if args.data != 'chembl':
        path_to_data = '/pool001/whgao/data/synth_net/st_' + args.rxn_template + '/st_' + args.data +'.json.gz'
        print('Reading data from ', path_to_data)
        sts = SyntheticTreeSet()
        sts.load(path_to_data)
        smis_query = [st.root.smiles for st in sts.sts]
        if args.num == -1:
            pass
        else:
            smis_query = smis_query[:args.num]
    else:
        df = pd.read_csv('/home/whgao/synth_net/chembl_20k.csv')
        smis_query = df.smiles.to_list()

    print('Start to decode!')

    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(predict.func, smis_query)

    smis_decoded = [r[0] for r in results]
    similaritys = [r[1] for r in results]
    trees = [r[2] for r in results]

    print("Finish decoding")
    print(f"Recovery rate {args.data}: {np.sum(np.array(similaritys) == 1.0) / len(similaritys)}")
    print(f"Average similarity {args.data}: {np.mean(np.array(similaritys))}")

    print('Saving ......')
    save_path = '../results/'
    df = pd.DataFrame({'query SMILES': smis_query, 'decode SMILES': smis_decoded, 'similarity': similaritys})
    df.to_csv(save_path + 'decode_result_' + args.data + '.csv.gz', compression='gzip', index=False)
    
    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(save_path + 'decoded_st_' + args.data + '.json.gz')

    print('Finish!')


