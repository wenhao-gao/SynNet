"""
This file contains a function to decode a single synthetic tree.
"""
import pandas as pd
import numpy as np
from syn_net.utils.data_utils import ReactionSet
from syn_net.utils.predict_utils import synthetic_tree_decoder_multireactant, load_modules_from_checkpoint, mol_fp
from pathlib import Path
from syn_net.config import DATA_PREPROCESS_DIR, DATA_EMBEDDINGS_DIR, CHECKPOINTS_DIR

# define some constants (here, for the Hartenfeller-Button test set)
nbits        = 4096
out_dim      = 256 # <=> morgan fingerprint with 256 bits
rxn_template = 'hb'
building_blocks_id = "enamine_us-2021-smiles"
featurize    = 'fp'
param_dir    = 'hb_fp_2_4096_256'
ncpu         = 1

# load the purchasable building block embeddings
file = Path(DATA_EMBEDDINGS_DIR) / f"{rxn_template}-{building_blocks_id}-embeddings.npy"
bb_emb = np.load(file)


# define paths to pretrained modules
path_to_act = Path(CHECKPOINTS_DIR) / f"{param_dir}/act.ckpt"
path_to_rt1 = Path(CHECKPOINTS_DIR) / f"{param_dir}/rt1.ckpt"
path_to_rxn = Path(CHECKPOINTS_DIR) / f"{param_dir}/rxn.ckpt"
path_to_rt2 = Path(CHECKPOINTS_DIR) / f"{param_dir}/rt2.ckpt"

# Load building blocks
building_blocks_file = Path(DATA_PREPROCESS_DIR) / f"{rxn_template}-{building_blocks_id}-matched.csv.gz"
building_blocks = pd.read_csv(building_blocks_file, compression='gzip')['SMILES'].tolist()
bb_dict         = {block: i for i,block in enumerate(building_blocks)} # dict is useful as lookup table for 2nd reactant during inference

# Load reaction templates
reaction_file = Path(DATA_PREPROCESS_DIR) / f"reaction-sets_{rxn_template}_{building_blocks_id}.json.gz"
rxn_set = ReactionSet()
rxn_set.load(reaction_file)
rxns    = rxn_set.rxns

# load the pre-trained modules
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

def func(smi):
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
    emb = mol_fp(smi)
    try:
        smi, similarity, tree, action = synthetic_tree_decoder_multireactant(
            z_target=emb,
            building_blocks=building_blocks,
            bb_dict=bb_dict,
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

    if action != 3:
        return None, 0, None
    else:
        return smi, similarity, tree
