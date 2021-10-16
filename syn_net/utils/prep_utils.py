"""
This file contains various utils for data preparation and preprocessing.
"""
import numpy as np
from scipy import sparse
from dgllife.model import load_pretrained
from tdc.chem_utils import MolConvert
from syn_net.utils.data_utils import SyntheticTree
import numpy as np
from syn_net.utils.data_utils import SyntheticTree, SyntheticTreeSet
from syn_net.utils.predict_utils import can_react, get_action_mask, get_reaction_mask, mol_fp, get_mol_embedding

def rdkit2d_embedding(smi):
    """
    Computes an embedding from the RDKit 2D descriptors.

    Args:
        smi (str): SMILES string.

    Returns:
        np.ndarray: A molecular embedding.
    """
    if smi is None:
        return np.zeros(200).reshape((-1, ))
    else:
        # define the RDKit 2D descriptor
        rdkit2d = MolConvert(src = 'SMILES', dst = 'RDKit2D')
        return rdkit2d(smi).reshape(-1, )


def organize(st, d_mol=300, target_embedding='fp', radius=2, nBits=4096, output_embedding='gin'):
    """
    Organizes the states and steps from the input synthetic tree into sparse matrices.

    Args:
        st (SyntheticTree): The input synthetic tree to organize.
        d_mol (int, optional): The molecular embedding size. Defaults to 300.
        target_embedding (str, optional): Indicates what kind of embedding to use
            for the input target (Morgan fingerprint --> 'fp' or GIN --> 'gin').
            Defaults to 'fp'.
        radius (int, optional): Morgan fingerprint radius to use. Defaults to 2.
        nBits (int, optional): Number of bits to use in the Morgan fingerprints.
            Defaults to 4096.
        output_embedding (str, optional): Indicates what type of embedding to use
            for the output node states. Defaults to 'gin'.

    Raises:
        ValueError: Raised if target embedding not supported.

    Returns:
        sparse.csc_matrix: Node states pulled from the tree.
        sparse.csc_matrix: Actions pulled from the tree.
    """
    # define model to use for molecular embedding
    model_type = 'gin_supervised_contextpred'
    device = 'cpu'
    model = load_pretrained(model_type).to(device)
    model.eval()

    states = []
    steps = []

    if output_embedding == 'gin':
        d_mol = 300
    elif output_embedding == 'fp_4096':
        d_mol = 4096
    elif output_embedding == 'fp_256':
        d_mol = 256
    elif output_embedding == 'rdkit2d':
        d_mol = 200

    if target_embedding == 'fp':
        target = mol_fp(st.root.smiles, radius, nBits).tolist()
    elif target_embedding == 'gin':
        target = get_mol_embedding(st.root.smiles, model=model).tolist()
    else:
        raise ValueError('Target embedding only supports fp and gin')

    most_recent_mol = None
    other_root_mol = None
    for i, action in enumerate(st.actions):

        most_recent_mol_embedding = mol_fp(most_recent_mol, radius, nBits).tolist()
        other_root_mol_embedding = mol_fp(other_root_mol, radius, nBits).tolist()
        state = most_recent_mol_embedding + other_root_mol_embedding + target

        if action == 3:
            step = [3] + [0] * d_mol + [-1] + [0] * d_mol + [0] * nBits

        else:
            r = st.reactions[i]
            mol1 = r.child[0]
            if len(r.child) == 2:
                mol2 = r.child[1]
            else:
                mol2 = None

            if output_embedding == 'gin':
                step = [action] + get_mol_embedding(mol1, model=model).tolist() + [r.rxn_id] + get_mol_embedding(mol2, model=model).tolist() + mol_fp(mol1, radius, nBits).tolist()
            elif output_embedding == 'fp_4096':
                step = [action] + mol_fp(mol1, 2, 4096).tolist() + [r.rxn_id] + mol_fp(mol2, 2, 4096).tolist() + mol_fp(mol1, radius, nBits).tolist()
            elif output_embedding == 'fp_256':
                step = [action] + mol_fp(mol1, 2, 256).tolist() + [r.rxn_id] + mol_fp(mol2, 2, 256).tolist() + mol_fp(mol1, radius, nBits).tolist()
            elif output_embedding == 'rdkit2d':
                step = [action] + rdkit2d_embedding(mol1).tolist() + [r.rxn_id] + rdkit2d_embedding(mol2).tolist() + mol_fp(mol1, radius, nBits).tolist()

        if action == 2:
            most_recent_mol = r.parent
            other_root_mol = None

        elif action == 1:
            most_recent_mol = r.parent

        elif action == 0:
            other_root_mol = most_recent_mol
            most_recent_mol = r.parent

        states.append(state)
        steps.append(step)

    return sparse.csc_matrix(np.array(states)), sparse.csc_matrix(np.array(steps))

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
