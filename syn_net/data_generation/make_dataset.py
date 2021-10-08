"""
This file generates synthetic tree data in a sequential fashion.
"""
import dill as pickle
import gzip
import pandas as pd
import numpy as np
import rdkit
from tqdm import tqdm

from synth_net.utils.data_utils import SyntheticTree, SyntheticTreeSet

# TODO add docstrings, types, clean up comments

def can_react(state, rxns):
    mol1 = state.pop()
    mol2 = state.pop()
    reaction_mask = [int(rxn.run_reaction([mol1, mol2]) is not None) for rxn in rxns]
    return sum(reaction_mask), reaction_mask

def get_action_mask(state, rxns):
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
        raise ValueError('Problem with state.')

def get_reaction_mask(smi, rxns):
    # Return all available reaction templates
    # List of available building blocks if 2
    # Exclude the case of len(available_list) == 0
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

def synthetic_tree_generator(building_blocks, reaction_templates, max_step=15):
    # Initialization
    tree = SyntheticTree()
    mol_recent = None

    # Start iteration
    try:
        for i in range(max_step):
            # Encode current state
            # from ipdb import set_trace; set_trace(context=11)
            state = tree.get_state() # a set
            # z_state = PMA(z_target, state)

            # Predict action type, masked selection
            # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
            # action_proba, z_mol1 = a_select(z_state)
            action_proba = np.random.rand(4)
            action_mask = get_action_mask(tree.get_state(), reaction_templates)
            action = np.argmax(action_proba * action_mask)

            # Select first molecule
            if action == 3:
                # End
                break
            elif action == 0:
                # Add
                # mol1 = NNsearch(z_mol1, building_blocks)
                mol1 = np.random.choice(building_blocks)
            else:
                # Expand or Merge
                mol1 = mol_recent

            # z_mol1 = GNN(mol1)

            # Select reaction
            # rxn_proba = a_rxn(z_state, z_mol1)
            reaction_proba = np.random.rand(len(reaction_templates))

            if action != 2:
                reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
            else:
                _, reaction_mask = can_react(tree.get_state(), reaction_templates)
                available_list = [[] for rxn in reaction_templates]

            if reaction_mask is None:
                if len(state) == 1:
                    action = 3
                    break
                else:
                    break

            rxn_id = np.argmax(reaction_proba * reaction_mask)
            rxn = reaction_templates[rxn_id]

            if rxn.num_reactant == 2:
                # Select second molecule
                if action == 2:
                    # Merge
                    temp = set(state) - set([mol1])
                    mol2 = temp.pop()
                else:
                    # Add or Expand
                    # z_mol2 = a_sele(z_state, z_mol1, rxn_type)
                    # mol2 = NNsearch(z_mol2, available_list)
                    mol2 = np.random.choice(available_list[rxn_id])
                # z_mol2 = GNN(mol2)
            else:
                mol2 = None

            # Run reaction
            mol_product = rxn.run_reaction([mol1, mol2])

            # Update
            tree.update(action, int(rxn_id), mol1, mol2, mol_product)
            mol_recent = mol_product
    except Exception as e:
        print(e)
        action = -1
        tree = None

    if action != 3:
        tree = None
    else:
        tree.update(action, None, None, None, None)

    return tree, action

if __name__ == '__main__':
    path_reaction_file = '/home/whgao/shared/Data/scGen/reactions_pis.pickle.gz'
    path_to_building_blocks = '/home/whgao/shared/Data/scGen/enamine_building_blocks_nochiral_matched.csv.gz'

    np.random.seed(6)

    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
    with gzip.open(path_reaction_file, 'rb') as f:
        rxns = pickle.load(f)

    Trial = 5
    num_finish = 0
    num_error = 0
    num_unfinish = 0

    trees = []
    for _ in tqdm(range(Trial)):
        tree, action = synthetic_tree_generator(building_blocks, rxns, max_step=15)
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

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save('st_data.json.gz')

    # data_file = gzip.open('st_data.pickle.gz', 'wb')
    # pickle.dump(trees, data_file)
    # data_file.close()
