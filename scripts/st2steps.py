"""
TODO
"""
from synth_net.utils.data_utils import *
import os
import gzip
import dill as pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from scipy import sparse

import rdkit
from rdkit import Chem, DataStructs
import rdkit.Chem.AllChem as AllChem

import torch
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from tdc.chem_utils import MolConvert
rdkit2d = MolConvert(src = 'SMILES', dst = 'RDKit2D')

model_type = 'gin_supervised_contextpred'
device = 'cpu'

model = load_pretrained(model_type).to(device)
model.eval()
readout = AvgPooling()

def graph_construction_and_featurization(smiles):
    """Construct graphs from SMILES and featurize them
    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation
    Returns
    -------
    list of DGLGraph
        List of graphs constructed and featurized
    list of bool
        Indicators for whether the SMILES string can be
        parsed by RDKit
    """
    graphs = []
    success = []
    for smi in tqdm(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)

    return graphs, success

def mol_embedding(smi, device='cpu'):
    if smi is None:
        return np.zeros(300)
    else:
        mol = Chem.MolFromSmiles(smi)
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
        return readout(bg, node_repr).detach().cpu().numpy().reshape(-1, )

def mol_fp(smi, radius=2, nBits=1024):
    if smi is None:
        return np.zeros(nBits)
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features

def rdkit2d_embedding(smi):
    if smi is None:
        return np.zeros(200).reshape((-1, ))
    else:
        return rdkit2d(smi).reshape(-1, )

def organize(st, d_mol=300, target_embedding='fp', radius=2, nBits=4096, output_embedding='gin'):
    
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
        target = mol_embedding(st.root.smiles).tolist()
    else:
        raise ValueError('Traget embedding only support fp and gin')

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
                step = [action] + mol_embedding(mol1).tolist() + [r.rxn_id] + mol_embedding(mol2).tolist() + mol_fp(mol1, radius, nBits).tolist()
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

