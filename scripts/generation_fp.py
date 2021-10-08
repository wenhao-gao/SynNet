"""
This file contains the code to generate synthetic trees, either with random 
choice or networks.
"""
from typing import Tuple, Union
import pandas as pd
import numpy as np
import rdkit
from tqdm import tqdm
import torch
from rdkit import Chem
from rdkit import DataStructs

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

    available_list = []  # available reactants for reacting with `smi`
    mol = rdkit.Chem.MolFromSmiles(smi)
    for i, rxn in enumerate(rxns):
        if reaction_mask[i] and rxn.num_reactant == 2:

            # identify the viable reactants if this is a bi-molecular reaction
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
model_type = 'gin_supervised_contextpred' # GIN used just for NN search
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
        fp = smi2fp(smi)
        return fp.reshape((1, -1))


bb_emb = np.load('/home/whgao/scGen/synth_net/data/enamine_us_emb.npy')
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
            # get the embedding for the first molecule
            e1 = _mol_embedding(state[0])
        except:
            import ipdb; ipdb.set_trace(context=9)
        if len(state) == 1:
            # there is no second molecule; set embedding to zero vector
            e2 = np.zeros((1, z_target.size))
        else:
            # get the embedding for the second molecule
            e2 = _mol_embedding(state[1])
        return np.concatenate([e1, e2, z_target], axis=1)

def synthetic_tree_decoder(z_target, 
                           building_blocks, 
                           bb_dict,
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
        z_state = set_embedding(z_target, state, mol_fp)

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

        # z_mol1 = get_mol_embedding(mol1, mol_embedder)
        z_mol1 = mol_fp(mol1)

        # Select reaction
        reaction_proba = rxn_net(torch.Tensor(np.concatenate([z_state, z_mol1], axis=1)))
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
                z_mol2 = reactant2_net(torch.Tensor(np.concatenate([z_state, z_mol1, one_hot_encoder(rxn_id, 91)], axis=1)))
                z_mol2 = z_mol2.detach().numpy()
                available = available_list[rxn_id]
                available = [bb_dict[available[i]] for i in range(len(available))]
                temp_emb = bb_emb[available]
                available_tree = KDTree(temp_emb, metric='euclidean')
                dist, ind = nn_search(z_mol2, _tree=available_tree)
                mol2 = building_blocks[available[ind]]
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
        tree = tree
    else:
        tree.update(act, None, None, None, None)

    return tree, act


if __name__ == '__main__':

    from synth_net.models.action import Action
    from synth_net.models.reactant1 import Reactant1
    from synth_net.models.rxn_fp import Rxn
    from synth_net.models.reactant2 import Reactant2

    path_to_reaction_file = '/home/whgao/scGen/synth_net/data/reactions_hb.json.gz'
    path_to_building_blocks = '/home/whgao/scGen/synth_net/data/enamine_us_matched.csv.gz'

    path_to_action = '/home/whgao/scGen/synth_net/synth_net/params/action_net.ckpt'
    path_to_reactant1 = '/home/whgao/scGen/synth_net/synth_net/params/rnt1_net.ckpt'
    path_to_rxn = '/home/whgao/scGen/synth_net/synth_net/params/rxn_net.ckpt'
    path_to_reactant2 = '/home/whgao/scGen/synth_net/synth_net/params/rnt2_net.ckpt'

    np.random.seed(6)

    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
    bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}

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

#     query_smis = ['Cc1cc(C)c(S(=O)(=O)Cl)c(C)c1CC(=O)N1CCN(C)c2ccc(-c3ccc(C(CC(=O)NCc4ccc(N(C)C)cc4)N(CCn4c(C)nc5cc(Cl)ccc54)C(=O)NCCCNC(=O)OC(C)(C)C)s3)cc21',
#              'CCN1CC(c2noc(Cn3c(-c4ccc(OCc5coc(-c6ccc(S(C)(=O)=O)cc6)n5)cc4)nc4c5c(c(Br)cc4c3=O)C(=O)c3ccccc3C5=O)n2)C(C(=O)C(C#N)c2ccccc2Br)C1=O',
#               'COc1cccc(CN(CCN(C)C)S(=O)(=O)CC2(F)CC(c3nc(-c4cccc(-c5nnc(CCl)o5)c4)n[nH]3)(C(N(CCCc3nnc(C)n3C)c3cc(Cl)nc(-c4ccoc4)n3)C(F)(F)CCn3ccnn3)C2)c1',
#                'Cc1ccccc1-c1noc(C=C(C2CNC2)C2(C(=O)NC3CCC4CN(C(=O)OC(C)(C)C)CC4C3)CC3(CC(N(C(=O)CN4CCS(=O)(=O)CC4)c4cc(C)c(S(N)(=O)=O)cc4[N+](=O)[O-])CCO3)C2)n1',
#                 'CCOC(=O)C(c1noc(-c2cc(S(=O)(=O)NCc3ccccc3OC)c(Cl)cc2Cl)n1)(C1CCOCC1)C(CCOCC1(C(=O)Oc2c(F)ccc3ccccc23)CCCN(C(=O)OC(C)(C)C)C1)CC(=O)C(C)(C)O',
#                  'COc1ccc(-c2ccccc2C2=Nc3ccc(C)cc3C(=O)N2c2ccc(-c3ccccc3)cc2)c2c(=O)n(-c3ccccn3)c(-c3ccccc3-c3ccc(-c4ccn(-c5cc(F)ccc5OCc5ccccn5)n4)cc3)nc12',
#                   'CCc1nc(CN2CCCC(N(C)C(=O)C(c3nnnn3C3CC4(C3)CC(F)(F)C4)N(C(=O)CC3(CCNC(=O)OC(C)(C)C)COCCO3)c3ccc(S(=O)(=O)N4CCN(C)CC4)cc3[N+](=O)[O-])C2)no1',
#                    'CC(=O)Nc1ccc(NC(=S)N2CCOCC2(Cc2ccccc2)CN(CCc2nc(-c3ccc(-c4ccccc4C(=O)Nc4ccc(-c5ccccc5)cc4)cc3)n[nH]2)S(=O)(=O)CC(F)(F)c2nc(-c3nonc3[N+](=O)[O-])n[nH]2)cc1',
#                     'CC#CCCCCC(O)(Cn1nnnc1CCN(C(=O)NC1CCCC(c2nc(C(=Cc3ccoc3)c3nc(CCNCC(F)(F)F)no3)n[nH]2)C1)c1ccc2c(c1)OCCO2)c1ccc2c(c1)CCC(=O)N2C']
# 
#     query_smis = ['COc1ccc(F)cc1CS(=O)(=O)Nc1ccc(C2CCC(N(C)C(=O)c3ccc(C#N)cc3)CC2)cc1',
#              'COC(=O)c1c(-c2ccccc2)csc1NC(=S)Nc1cc(S(=O)(=O)N2CCCC2)ccc1OCC(F)(F)F',
#               'CCOc1ccc(OCC)c(NC(=S)N(Cc2ccccc2)c2nnc(SCC(=O)NCC3CCCCC3)s2)c1',
#                'COc1cc(OCCC2CCCN2C)ccc1NC(=O)Nc1ccc(I)c(Cl)c1',
#                 'O=C(Nc1cnn2ccc(-c3n~c4cc(C(=O)O)ccc4o3)cc12)OCC1c2ccccc2-c2ccccc21',
#                  'COCC(C)NS(=O)(=O)c1ccc(N(C(C)C)C2CCCN(C(=O)OC(C)(C)C)CC2)c([N+](=O)[O-])c1',
#                   'CCN(CC)CCn1c(-c2cnc(N3CCOC(CNC(=O)OC(C)(C)C)C3)cn2)n~c2cc([N+](=O)[O-])ccc21',
#                    'CCOC(=O)C(=CNc1ccccc1CC)c1noc(-c2ccc(C(=O)Nc3ccccc3NC(=O)OC(C)(C)C)cc2)n1',
#                     'CN(C)S(=O)(=O)c1ccc(CCS(=O)(=O)NCc2cn(-c3ccccc3)nc2-c2ccncc2)cc1']
#
    query_smis = ['OC1=CC(C(O)CNC)=CC=C1O']

    Trial = len(query_smis)
    num_finish = 0
    num_error = 0
    num_unfinish = 0

    trees = []
    for _ in tqdm(range(Trial)):
        query_smi = query_smis[_]
        print(f'The query smiles is: {query_smi}')
        z_target = mol_fp(query_smi)
        tree, action = synthetic_tree_decoder(z_target, 
                                              building_blocks,
                                              bb_dict,
                                              rxns, 
                                              mol_embedder, 
                                              action_net, 
                                              reactant1_net, 
                                              rxn_net, 
                                              reactant2_net, 
                                              max_step=20)
        if action == 3:
            trees.append(tree)
            num_finish += 1
        elif action == -1:
            num_error += 1
        else:
            num_unfinish += 1
            trees.append(tree)

        tree._print()
        ms = [Chem.MolFromSmiles(sm) for sm in [query_smi, tree.root.smiles]]
        fps = [Chem.RDKFingerprint(x) for x in ms]
        print('Tanimoto similarity is: ', DataStructs.FingerprintSimilarity(fps[0],fps[1]))

    print('Total trial: ', Trial)
    print('num of finished trees: ', num_finish)
    print('num of unfinished tree: ', num_unfinish)
    print('num of error processes: ', num_error)
