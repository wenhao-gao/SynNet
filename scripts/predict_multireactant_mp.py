"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from syn_net.config import (CHECKPOINTS_DIR, DATA_EMBEDDINGS_DIR,
                            DATA_PREPARED_DIR, DATA_PREPROCESS_DIR,
                            DATA_RESULT_DIR)
from syn_net.utils.data_utils import ReactionSet, SyntheticTreeSet
from syn_net.utils.predict_utils import (load_modules_from_checkpoint, mol_fp,
                                         synthetic_tree_decoder_multireactant)

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

def _fetch_reaction_templates(file: str):
    # Load reaction templates
    rxn_set = ReactionSet().load(file)
    return rxn_set.rxns

def _fetch_building_blocks_embeddings(file: str):
    """Load the purchasable building block embeddings."""
    return np.load(file)

def _fetch_building_blocks(file: str):
    """Load the building blocks"""
    return pd.read_csv(file, compression='gzip')['SMILES'].tolist()

def _load_pretrained_model(path_to_checkpoints: str):
    """Wrapper to load modules from checkpoint."""
    # Define paths to pretrained models.
    path_to_act = Path(path_to_checkpoints) / "act.ckpt"
    path_to_rt1 = Path(path_to_checkpoints) / "rt1.ckpt"
    path_to_rxn = Path(path_to_checkpoints) / "rxn.ckpt"
    path_to_rt2 = Path(path_to_checkpoints) / "rt2.ckpt"

    # Load the pre-trained models.
    act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
        path_to_act=path_to_act,
        path_to_rt1=path_to_rt1,
        path_to_rxn=path_to_rxn,
        path_to_rt2=path_to_rt2,
        featurize=featurize,
        rxn_template=rxn_template,
        out_dim=out_dim,
        nbits=nbits,
        ncpu=ncpu,
    )
    return act_net, rt1_net, rxn_net, rt2_net

def func(smiles: str):
    """
    Generates the synthetic tree for the input molecular embedding.

    Args:
        smi (str): SMILES string corresponding to the molecule to decode.

    Returns:
        smi (str): SMILES for the final chemical node in the tree.
        similarity (float): Similarity measure between the final chemical node
            and the input molecule.
        tree (SyntheticTree): The generated synthetic tree.
    """
    emb = mol_fp(smiles)
    try:
        smi, similarity, tree, action = synthetic_tree_decoder_multireactant(
            z_target=emb,
            building_blocks=building_blocks,
            bb_dict=building_blocks_dict,
            reaction_templates=rxns,
            mol_embedder=mol_fp,
            action_net=act_net,
            reactant1_net=rt1_net,
            rxn_net=rxn_net,
            reactant2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template=rxn_template,
            n_bits=nbits,
            beam_width=3,
            max_step=15)
    except Exception as e:
        print(e)
        action = -1

    if action != 3: # aka tree has not been properly ended
        smi = None
        similarity =  0
        tree = None
    
    return smi, similarity, tree


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("--radius", type=int, default=2,
                            help="Radius for Morgan Fingerprint")
    parser.add_argument("-b", "--nbits", type=int, default=4096,
                            help="Number of Bits for Morgan Fingerprint")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("--ncpu", type=int, default=1,
                        help="Number of cpus")
    parser.add_argument("-n", "--num", type=int, default=1,
                        help="Number of molecules to predict.")
    parser.add_argument("-d", "--data", type=str, default='test',
                        help="Choose from ['train', 'valid', 'test', 'chembl']")
    parser.add_argument("-o", "--outputembedding", type=str, default='fp_256',
                        help="Choose from ['fp_4096', 'fp_256', 'gin', 'rdkit2d']")
    args = parser.parse_args()

    nbits        = args.nbits
    out_dim      = args.outputembedding.split("_")[-1] # <=> morgan fingerprint with 256 bits
    rxn_template = args.rxn_template
    building_blocks_id = "enamine_us-2021-smiles"
    featurize    = args.featurize
    radius       = args.radius
    ncpu         = args.ncpu
    param_dir    = f"{rxn_template}_{featurize}_{radius}_{nbits}_{out_dim}" 

    # Load data ...
    # ... query molecules (i.e. molecules to decode)
    smiles_queries = _fetch_data(args.data)
    if args.num > 0: # Select only n queries
        smiles_queries = smiles_queries[:args.num]

    # ... building blocks
    file = Path(DATA_PREPROCESS_DIR) / f"{rxn_template}-{building_blocks_id}-matched.csv.gz"
    building_blocks = _fetch_building_blocks(file)
    building_blocks_dict = {block: i for i,block in enumerate(building_blocks)} # dict is used as lookup table for 2nd reactant during inference

    # ... reaction templates
    file = Path(DATA_PREPROCESS_DIR) / f"reaction-sets_{rxn_template}_{building_blocks_id}.json.gz"
    rxns = _fetch_reaction_templates(file)

    # ... building blocks
    file = Path(DATA_EMBEDDINGS_DIR) / f"{rxn_template}-{building_blocks_id}-embeddings.npy"
    bb_emb = _fetch_building_blocks_embeddings(file)

    # ... models
    path = Path(CHECKPOINTS_DIR) / f"{param_dir}"
    act_net, rt1_net, rxn_net, rt2_net = _load_pretrained_model(path)


    # Decode queries, i.e. the target molecules.
    print(f'Start to decode {len(smiles_queries)} target molecules.')
    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(func, smiles_queries)
    print('Finished decoding.')


    # Print some results from the prediction
    smis_decoded = [r[0] for r in results]
    similarities = [r[1] for r in results]
    trees        = [r[2] for r in results]

    recovery_rate = (np.asfarray(similarities)==1.0).sum()/len(similarities)
    avg_similarity = np.mean(similarities)
    print(f"For {args.data}:")
    print(f"  {recovery_rate=}")
    print(f"  {avg_similarity=}")

    # Save to local dir
    print('Saving results to {DATA_RESULT_DIR} ...')
    df = pd.DataFrame({'query SMILES' : smiles_queries, 
                       'decode SMILES': smis_decoded, 
                       'similarity'   : similarities})
    df.to_csv(f'{DATA_RESULT_DIR}/decode_result_{args.data}.csv.gz', 
              compression='gzip', 
              index=False,)
    
    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(f'{DATA_RESULT_DIR}/decoded_st_{args.data}.json.gz')

    print('Finish!')
