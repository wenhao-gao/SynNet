"""
Splits a synthetic tree into states and steps.
"""
from syn_net.utils.data_utils import *
import os
import numpy as np
from tqdm import tqdm
from scipy import sparse
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from tdc.chem_utils import MolConvert
from syn_net.utils.predict_utils import get_mol_embedding, mol_fp


# define the RDKit 2D descriptor
rdkit2d = MolConvert(src = 'SMILES', dst = 'RDKit2D')

# define model to use for molecular embedding
model_type = 'gin_supervised_contextpred'
device = 'cpu'
model = load_pretrained(model_type).to(device)
model.eval()
readout = AvgPooling()


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
        target = get_mol_embedding(st.root.smiles).tolist()
    else:
        raise ValueError('Traget embedding only supports fp and gin')

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
                step = [action] + get_mol_embedding(mol1).tolist() + [r.rxn_id] + get_mol_embedding(mol2).tolist() + mol_fp(mol1, radius, nBits).tolist()
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

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numbersave", type=int, default=999999999999,
                        help="Save number")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Increase output verbosity")
    parser.add_argument("-e", "--targetembedding", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-o", "--outputembedding", type=str, default='gin',
                        help="Choose from ['fp_4096', 'fp_256', 'gin', 'rdkit2d']")
    parser.add_argument("-r", "--radius", type=int, default=2,
                        help="Radius for Morgan Fingerprint")
    parser.add_argument("-b", "--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan Fingerprint")
    parser.add_argument("-d", "--datasettype", type=str, default='train',
                        help="Choose from ['train', 'valid', 'test']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    args = parser.parse_args()

    dataset_type = args.datasettype
    embedding = args.targetembedding
    path_st = '/pool001/whgao/data/synth_net/st_hb/st_' + dataset_type + '.json.gz'
    save_dir = '/pool001/whgao/data/synth_net/hb_' + embedding + '_' + str(args.radius) + '_' + str(args.nbits) + '_' + str(args.outputembedding) + '/'

    st_set = SyntheticTreeSet()
    st_set.load(path_st)
    print('Original length: ', len(st_set.sts))
    data = st_set.sts
    del st_set
    print('Working length: ', len(data))

    states = []
    steps = []

    num_save = args.numbersave
    idx = 0
    save_idx = 0
    for st in tqdm(data):
        try:
            state, step = organize(st, target_embedding=embedding, radius=args.radius, nBits=args.nbits, output_embedding=args.outputembedding)
        except Exception as e:
            print(e)
            continue
        states.append(state)
        steps.append(step)
        idx += 1
        if idx % num_save == 0:
            print('Saving......')
            states = sparse.vstack(states)
            steps = sparse.vstack(steps)
            sparse.save_npz(save_dir + 'states_' + str(save_idx) + '_' + dataset_type + '.npz', states)
            sparse.save_npz(save_dir + 'steps_' + str(save_idx) + '_' + dataset_type + '.npz', steps)
            save_idx += 1
            del states
            del steps
            states = []
            steps = []

    del data

    if len(steps) != 0:
        states = sparse.vstack(states)
        steps = sparse.vstack(steps)

        print('Saving......')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        sparse.save_npz(save_dir + 'states_' + str(save_idx) + '_' + dataset_type + '.npz', states)
        sparse.save_npz(save_dir + 'steps_' + str(save_idx) + '_' + dataset_type + '.npz', steps)

    print('Finish!')
