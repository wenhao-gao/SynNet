"""
This file contains functions for learning graph embeddings using GIN from SMILES.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple
import multiprocessing as mp

import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs

import torch
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer

from tdc.chem_utils import MolConvert
rdkit2d = MolConvert(src = 'SMILES', dst = 'RDKit2D')

model_type = 'gin_supervised_contextpred'
device = 'cpu'

model = load_pretrained(model_type).to(device) # used to learn embedding
model.eval()
readout = AvgPooling()  # TODO would not be better if this was sum?

def graph_construction_and_featurization(smiles : list) -> Tuple[list, list]:
    """Constructs graphs from SMILES and featurizes them.

    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation

    Returns
    -------
    graphs : list of DGLGraph
        List of graphs constructed and featurized
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

def mol_embedding(smi : str, device : str='cuda:0') -> torch.Tensor:
    """Constructs a graph embedding for an input SMILES.

    Parameters
    ----------
    smi : str
        A SMILES string
    device : str
        Indicates the device to run on. Default 'cuda:0'

    Returns
    -------
    torch.Tensor
        The graph embedding
    """
    if smi is None:
        return np.zeros(300)
    else:
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
        return readout(bg, node_repr).detach().cpu().numpy().reshape(-1, ).tolist()


def fp_embedding(smi, _radius=2, _nBits=4096):
    if smi is None:
        return np.zeros(_nBits).reshape((-1, )).tolist()
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape((-1, )).tolist()

def fp_4096(smi):
    return fp_embedding(smi, _radius=2, _nBits=4096)

def fp_2048(smi):
    return fp_embedding(smi, _radius=2, _nBits=2048)

def fp_1024(smi):
    return fp_embedding(smi, _radius=2, _nBits=1024)

def fp_512(smi):
    return fp_embedding(smi, _radius=2, _nBits=512)

def fp_256(smi):
    return fp_embedding(smi, _radius=2, _nBits=256)

def rdkit2d_embedding(smi):
    if smi is None:
        return np.zeros(200).reshape((-1, )).tolist()
    else:
        return rdkit2d(smi).tolist()


def get_mol_embedding_func(feature):
    if feature == 'gin':
        embedding_func = lambda smi: gin_embedding(smi, device='cpu')
    elif feature == 'fp_4096':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=4096)
    elif feature == 'fp_2048':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=2048)
    elif feature == 'fp_1024':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=1024)
    elif feature == 'fp_512':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=512)
    elif feature == 'fp_256':
        embedding_func = lambda smi: fp_embedding(smi, _nBits=256)
    elif feature == 'rdkit2d':
        embedding_func = rdkit2d_embedding
    return embedding_func

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="gin",
                                        help="Objective function to optimize")
    parser.add_argument("--ncpu", type=int, default=16,
                                    help="Number of cpus")
    args = parser.parse_args()

    path = '/pool001/whgao/data/synth_net/st_hb/'
    data = pd.read_csv(path + 'enamine_us_matched.csv.gz', compression='gzip')['SMILES'].tolist()
    print('Total data: ', len(data))

    embeddings = []
    for smi in tqdm(data):
        embeddings.append(gin_embedding(smi))

    embedding = np.array(embeddings)
    np.save(path + 'enamine_us_emb_' + args.feature + '.npy', embeddings)

    print('Finish!')

