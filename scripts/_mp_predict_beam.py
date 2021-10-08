"""
This file contains a function to decode a single synthetic tree
"""
import pandas as pd
import numpy as np
from synth_net.utils.data_utils import ReactionSet
from dgllife.model import load_pretrained
from syn_net.utils.predict_utils import tanimoto_similarity, load_modules_from_checkpoint, mol_fp
from syn_net.utils.predict_beam_utils import synthetic_tree_decoder


# define some constants (here, for the Hartenfeller-Button test set)
nbits = 4096
out_dim = 300
rxn_template = 'hb'
featurize = 'fp'
param_dir = 'hb_fp_2_4096'
ncpu = 16

# define model to use for molecular embedding
model_type = 'gin_supervised_contextpred'
device = 'cpu'
mol_embedder = load_pretrained(model_type).to(device)
mol_embedder.eval()

# load the purchasable building block embeddings
bb_emb = np.load('/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy')

# define path to the reaction templates and purchasable building blocks
path_to_reaction_file = '/pool001/whgao/data/synth_net/st_' + rxn_template + '/reactions_' + rxn_template + '.json.gz'
path_to_building_blocks = '/pool001/whgao/data/synth_net/st_' + rxn_template + '/enamine_us_matched.csv.gz'

# define paths to pretrained modules
param_path = '/home/whgao/scGen/synth_net/synth_net/params/' + param_dir + '/'
path_to_act = param_path + 'act.ckpt'
path_to_rt1 = param_path + 'rt1.ckpt'
path_to_rxn = param_path + 'rxn.ckpt'
path_to_rt2 = param_path + 'rt2.ckpt'

# load the purchasable building block SMILES to a dictionary
building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}

# load the reaction templates as a ReactionSet object
rxn_set = ReactionSet()
rxn_set.load(path_to_reaction_file)
rxns = rxn_set.rxns

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
    Generates the synthetic tree for the input moleular string.

    Args:
        smi (str): Molecule (SMILES) to decode.

    Returns:
        np.ndarray or None: State of the generated synthetic tree.
        float: The best score.
        SyntheticTree: The generated synthetic tree.
    """
    emb = mol_fp(smi)
    try:
        tree, action = synthetic_tree_decoder(emb, building_blocks, bb_dict, rxns, mol_embedder, act_net, rt1_net, rxn_net, rt2_net, bb_emb, beam_width=10, rxn_template=rxn_template, n_bits=nbits, max_step=15)
    except Exception as e:
        print(e)
        action = -1

    # tree, action = synthetic_tree_decoder(emb, building_blocks, bb_dict, rxns, mol_embedder, act_net, rt1_net, rxn_net, rt2_net, max_step=15)

    # import ipdb; ipdb.set_trace(context=9)
    # tree._print()
    # print(action)
    # print(np.max(oracle(tree.get_state())))
    # print()

    if action != 3:
        return None, 0, None
    else:
        scores = tanimoto_similarity(emb, tree.get_state())
        max_score_idx = np.where(scores == np.max(scores))[0][0]
        return tree.get_state()[max_score_idx], np.max(scores), tree
