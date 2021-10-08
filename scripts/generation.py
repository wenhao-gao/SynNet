"""
This file contains the code to generate a synthetic tree, either with random 
choice or networks
"""
from typing import Tuple, Union
import pandas as pd
import numpy as np
import rdkit
from tqdm import tqdm
import torch
from rdkit import Chem

from synth_net.utils.data_utils import Reaction, ReactionSet, SyntheticTree, SyntheticTreeSet
from sklearn.neighbors import KDTree

import dgl
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from tdc.chem_utils import MolConvert
import shutup
shutup.please()


def can_react(state : np.ndarray, rxns : list) -> Tuple[np.ndarray, np.ndarray]:
    """Determines if two molecules can react using any of the input reactions.

    Parameters
    ----------
    state : np.ndarray
        The current state in the synthetic tree.
    rxns : list of Reaction objects
        Contains available reaction templates.

    Returns
    -------
    np.ndarray
        The sum of the reaction mask tells us how many reactions are viable for 
        the two molecules.
    np.ndarray
        The reaction mask. Masks out reactions which are not viable for the two 
        molecules.
    """
    mol1 = state.pop()
    mol2 = state.pop()
    reaction_mask = [int(rxn.run_reaction([mol1, mol2]) is not None) for rxn in rxns]
    return sum(reaction_mask), reaction_mask

def get_action_mask(state : np.ndarray, rxns : list) -> np.ndarray:
    """Determines which actions can apply to a given state in the synthetic tree.

    Parameters
    ----------
    state : np.ndarray
        The current state in the synthetic tree.
    rxns : list of Reaction objects
        Contains available reaction templates.

    Returns
    -------
    np.ndarray
        The action mask. Masks out unviable actions from the current state.
    """
    # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
    if len(state) == 0:
        return np.array([1, 0, 0, 0])
    elif len(state) == 1:
        return np.array([1, 1, 0, 1])
    elif len(state) == 2:
        can_react_, _ = can_react(state, rxns)
        if can_react_:
            return np.array([0, 1, 1, 0])
        else:
            return np.array([0, 1, 0, 0])
    else:
        import ipdb; ipdb.set_trace(context=9)
        raise ValueError('Problem with state.')

def get_reaction_mask(smi : str, rxns : list) -> Tuple[Union[list, None], Union[list, None]]:
    """Determines which reaction templates can apply to the input molecule.

    Parameters
    ----------
    smi : str
        The SMILES string corresponding to the molecule in question.
    rxns : list of Reaction objects
        Contains available reaction templates.

    Returns
    -------
    reaction_mask : list of ints, or None
        The reaction template mask. Masks out reaction templates which are not 
        viable for the input molecule. If there are no viable reaction templates
        identified, is simply None.
    available_list : list of TODO, or None
        Contains available reactants if at least one viable reaction  template is 
        identified. Else is simply None.
    """
    reaction_mask = [int(rxn.is_reactant(smi)) for rxn in rxns]

    if sum(reaction_mask) == 0:
        return None, None
    available_list = []
    mol = rdkit.Chem.MolFromSmiles(smi)
    for i, rxn in enumerate(rxns):
        if reaction_mask[i] and rxn.num_reactant == 2:

            if rxn.is_reactant_first(mol):
                available_list.append(rxn.available_reactants[1])
            elif rxn.is_reactant_second(mol):
                available_list.append(rxn.available_reactants[0])
            else:
                raise ValueError('Check the reactants')

            if len(available_list[-1]) == 0:
                reaction_mask[i] = 0

        else:
            available_list.append([])

    return reaction_mask, available_list

def graph_construction_and_featurization(smiles : list) -> Tuple[list, list]:
    """Constructs graphs from SMILES and featurizes them.

    Parameters
    ----------
    smiles : list of str
        SMILES of molecules to embed

    Returns
    -------
    graphs : list of DGLGraph
        List of featurized graphs which were constructed
    success : list of bool
        Indicators for whether the SMILES string can be parsed by RDKit
    """
    graphs = []
    success = []
    for smi in tqdm(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            # convert RDKit.Mol into featurized bi-directed DGLGraph 
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)

    return graphs, success
    
def one_hot_encoder(dim : int, space : int) -> np.ndarray:
    """Returns one-hot encoded vector of length=`space` containing a one at the 
    specified `dim` and zero elsewhere.
    """
    vec = np.zeros((1, space))
    vec[0, dim] = 1
    return vec

readout = AvgPooling()
model_type = 'gin_supervised_contextpred'
device = 'cuda:0'
mol_embedder = load_pretrained(model_type).to(device)

def get_mol_embedding(smi : str, 
                      model : dgllife.model=mol_embedder, 
                      device : str='cuda:0', 
                      readout : dgl.nn.pytorch.glob=readout) -> torch.Tensor:
    """Computes the molecular graph embedding for the input SMILES.

    Parameters
    ----------
    smi : str
        SMILES of molecule to embed
    model : dgllife.model
        Pre-trained NN model to use for computing the embedding. Default GIN
    device : str
        Indicates the device to run on. Default 'cuda:0'
    readout : dgl.nn.pytorch.glob
        Readout function to use for computing the graph embedding

    Returns
    -------
    torch.Tensor
        Learned embedding for input molecule
    """
    mol = Chem.MolFromSmiles(smi)
    # convert RDKit.Mol into featurized bi-directed DGLGraph 
    g = mol_to_bigraph(mol, add_self_loop=True,
                       node_featurizer=PretrainAtomFeaturizer(),
                       edge_featurizer=PretrainBondFeaturizer(),
                       canonical_atom_order=False)
    bg = g.to(device)
    nfeats = [bg.ndata.pop('atomic_number').to(device),
              bg.ndata.pop('chirality_type').to(device)]
    efeats = [bg.edata.pop('bond_type').to(device),
              bg.edata.pop('bond_direction_type').to(device)]
    with torch.no_grad():
        node_repr = model(bg, nfeats, efeats)
    return readout(bg, node_repr).detach().cpu().numpy()

smi2fp = MolConvert(src = 'SMILES', dst = 'Morgan')
def mol_fp(smi : str) -> np.ndarray:
    """Returns the Morgan fingerprint for the molecule given by input SMILES.
    """
    if smi is None:
        return np.zeros(1024)
    else:
        fp = smi2fp(smi, radius=2, nBits=1024)
        return fp


bb_emb = np.load('/home/whgao/scGen/synthetic_tree_generation/data/enamine_us_emb.npy')
kdtree = KDTree(bb_emb, metric='euclidean')
def nn_search(_e, _tree=kdtree, _k=1):
    """Conducts a nearest-neighbors search. TODO types
    """
    dist, ind = _tree.query(_e, k=_k)
    return dist[0][0], ind[0][0]

def set_embedding(z_target : np.ndarray, 
                  state : np.ndarray, 
                  _mol_embedding=get_mol_embedding) -> np.ndarray:
    """Computes embeddings for all molecules in input state.
    TODO add params/returns to docstring
    """
    if len(state) == 0:
        return np.concatenate([np.zeros((1, 2 * z_target.size)), z_target], axis=1)
    else:
        try:
            e1 = _mol_embedding(state[0])
        except:
            import ipdb; ipdb.set_trace(context=9)
        if len(state) == 1:
            e2 = np.zeros((1, z_target.size))
        else:
            e2 = _mol_embedding(state[1])
        return np.concatenate([e1, e2, z_target], axis=1)

def synthetic_tree_decoder(z_target, 
                           building_blocks, 
                           reaction_templates, 
                           mol_embedder, 
                           action_net, 
                           reactant1_net, 
                           rxn_net, 
                           reactant2_net, 
                           max_step=15):
    """TODO add description/params/returns to docstring; missing types
    """
    # Initialization
    tree = SyntheticTree()
    mol_recent = None

    # Start iteration
    # try:
    for i in range(max_step):
        # Encode current state
        # from ipdb import set_trace; set_trace(context=11)
        state = tree.get_state() # a set
        z_state = set_embedding(z_target, state)

        # Predict action type, masked selection
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        action_proba = action_net(torch.Tensor(z_state)) 
        action_proba = action_proba.squeeze().detach().numpy() + 0.0001
        action_mask = get_action_mask(tree.get_state(), reaction_templates)
        act = np.argmax(action_proba * action_mask)

        # import ipdb; ipdb.set_trace(context=9)
        
        z_mol1 = reactant1_net(torch.Tensor(np.concatenate([z_state, one_hot_encoder(act, 4)], axis=1)))
        z_mol1 = z_mol1.detach().numpy()

        # Select first molecule
        if act == 3:
            # End
            break
        elif act == 0:
            # Add
            dist, ind = nn_search(z_mol1)
            mol1 = building_blocks[ind]
        else:
            # Expand or Merge
            mol1 = mol_recent

        z_mol1 = get_mol_embedding(mol1, mol_embedder)

        # Select reaction
        reaction_proba = rxn_net(torch.Tensor(np.concatenate([z_state, one_hot_encoder(act, 4), z_mol1], axis=1)))
        reaction_proba = reaction_proba.squeeze().detach().numpy() + 0.0001

        if act != 2:
            reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
        else:
            _, reaction_mask = can_react(tree.get_state(), reaction_templates)
            available_list = [[] for rxn in reaction_templates]

        if reaction_mask is None:
            if len(state) == 1:
                act = 3
                break
            else:
                break

        rxn_id = np.argmax(reaction_proba * reaction_mask)
        rxn = reaction_templates[rxn_id]

        if rxn.num_reactant == 2:
            # Select second molecule
            if act == 2:
                # Merge
                temp = set(state) - set([mol1])
                mol2 = temp.pop()
            else:
                # Add or Expand
                z_mol2 = reactant2_net(torch.Tensor(np.concatenate([z_state, one_hot_encoder(act, 4), z_mol1, one_hot_encoder(rxn_id, 4700)], axis=1)))
                z_mol2 = z_mol2.detach().numpy()
                dist, ind = nn_search(z_mol2)
                mol2 = building_blocks[ind]
        else:
            mol2 = None

        # Run reaction
        mol_product = rxn.run_reaction([mol1, mol2])
        if mol_product is None:
            if len(tree.get_state()) == 1:
                act = 3
                break
            else:
                break

        # Update
        tree.update(act, int(rxn_id), mol1, mol2, mol_product)
        mol_recent = mol_product
    # except Exception as e:
    #     print(e)
    #     act = -1
    #     tree = None

    if act != 3:
        tree = None
    else:
        tree.update(act, None, None, None, None)

    return tree, act


if __name__ == '__main__':

    from synth_net.models.action import Action
    from synth_net.models.reactant1 import Reactant1
    from synth_net.models.rxn import Rxn
    from synth_net.models.reactant2 import Reactant2

    path_to_reaction_file = '/home/whgao/scGen/synth_net/data/reactions_hb.json.gz'
    path_to_building_blocks = '/home/whgao/scGen/synth_net/data/enamine_us_matched.csv.gz'

    path_to_action = '/home/whgao/scGen/synth_net/synth_net/params/action_net.ckpt'
    path_to_reactant1 = '/home/whgao/scGen/synth_net/synth_net/params/rnt1_net.ckpt'
    path_to_rxn = '/home/whgao/scGen/synth_net/synth_net/params/rxn_net.ckpt'
    path_to_reactant2 = '/home/whgao/scGen/synth_net/synth_net/params/rnt2_net.ckpt'

    np.random.seed(6)

    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()

    rxn_set = ReactionSet()
    rxn_set.load(path_to_reaction_file)
    rxns = rxn_set.rxns
    # with gzip.open(path_reaction_file, 'rb') as f:
    #     rxns = pickle.load(f)

    action_net = Action.load_from_checkpoint(path_to_action)
    reactant1_net = Reactant1.load_from_checkpoint(path_to_reactant1)
    rxn_net = Rxn.load_from_checkpoint(path_to_rxn)
    reactant2_net = Reactant2.load_from_checkpoint(path_to_reactant2)

    action_net.eval()
    reactant1_net.eval()
    rxn_net.eval()
    reactant2_net.eval()

    query_smi = 'Cn1ccc(N(CCc2ccccc2N)c2ccccc2N)n1'
    print(f'The query smiles is: {query_smi}')
    z_target = get_mol_embedding(query_smi, mol_embedder, device)
    # z_target = mol_fp(query_smi)

    Trial = 5
    num_finish = 0
    num_error = 0
    num_unfinish = 0

    trees = []
    for _ in tqdm(range(Trial)):
        tree, action = synthetic_tree_decoder(z_target, 
                                              building_blocks, 
                                              rxns, 
                                              mol_embedder, 
                                              action_net, 
                                              reactant1_net, 
                                              rxn_net, 
                                              reactant2_net, 
                                              max_step=15)
        if action == 3:
            trees.append(tree)
            num_finish += 1
        elif action == -1:
            num_error += 1
        else:
            num_unfinish += 1

    print('Total trial: ', Trial)
    print('num of finished trees: ', num_finish)
    print('num of unfinished tree: ', num_unfinish)
    print('num of error processes: ', num_error)

    for t in trees:
        t.print()
    
    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save('generated_st.json.gz')
