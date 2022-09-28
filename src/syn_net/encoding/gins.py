import functools

import numpy as np
import torch
import tqdm
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, mol_to_bigraph
from rdkit import Chem


@functools.lru_cache(1)
def _fetch_gin_pretrained_model(model_name: str):
    """Get a GIN pretrained model to use for creating molecular embeddings"""
    device = "cpu"
    model = load_pretrained(model_name).to(device)  # used to learn embedding
    model.eval()
    return model


def graph_construction_and_featurization(smiles):
    """
    Constructs graphs from SMILES and featurizes them.

    Args:
        smiles (list of str): Contains SMILES of molecules to embed.

    Returns:
        graphs (list of DGLGraph): List of graphs constructed and featurized.
        success (list of bool): Indicators for whether the SMILES string can be
            parsed by RDKit.
    """
    graphs = []
    success = []
    for smi in tqdm(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(
                mol,
                add_self_loop=True,
                node_featurizer=PretrainAtomFeaturizer(),
                edge_featurizer=PretrainBondFeaturizer(),
                canonical_atom_order=False,
            )
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)

    return graphs, success


def mol_embedding(smi, device="cpu", readout=AvgPooling()):
    """
    Constructs a graph embedding using the GIN network for an input SMILES.

    Args:
        smi (str): A SMILES string.
        device (str): Indicates the device to run on ('cpu' or 'cuda:0'). Default 'cpu'.

    Returns:
        np.ndarray: Either a zeros array or the graph embedding.
    """
    name = "gin_supervised_contextpred"
    gin_pretrained_model = _fetch_gin_pretrained_model(name)

    # get the embedding
    if smi is None:
        return np.zeros(300)
    else:
        mol = Chem.MolFromSmiles(smi)
        # convert RDKit.Mol into featurized bi-directed DGLGraph
        g = mol_to_bigraph(
            mol,
            add_self_loop=True,
            node_featurizer=PretrainAtomFeaturizer(),
            edge_featurizer=PretrainBondFeaturizer(),
            canonical_atom_order=False,
        )
        bg = g.to(device)
        nfeats = [
            bg.ndata.pop("atomic_number").to(device),
            bg.ndata.pop("chirality_type").to(device),
        ]
        efeats = [
            bg.edata.pop("bond_type").to(device),
            bg.edata.pop("bond_direction_type").to(device),
        ]
        with torch.no_grad():
            node_repr = gin_pretrained_model(bg, nfeats, efeats)
        return (
            readout(bg, node_repr)
            .detach()
            .cpu()
            .numpy()
            .reshape(
                -1,
            )
            .tolist()
        )


def get_mol_embedding(smi, model, device="cpu", readout=AvgPooling()):
    """
    Computes the molecular graph embedding for the input SMILES.

    Args:
        smi (str): SMILES of molecule to embed.
        model (dgllife.model, optional): Pre-trained NN model to use for
            computing the embedding.
        device (str, optional): Indicates the device to run on. Defaults to 'cpu'.
        readout (dgl.nn.pytorch.glob, optional): Readout function to use for
            computing the graph embedding. Defaults to readout.

    Returns:
        torch.Tensor: Learned embedding for the input molecule.
    """
    mol = Chem.MolFromSmiles(smi)
    g = mol_to_bigraph(
        mol,
        add_self_loop=True,
        node_featurizer=PretrainAtomFeaturizer(),
        edge_featurizer=PretrainBondFeaturizer(),
        canonical_atom_order=False,
    )
    bg = g.to(device)
    nfeats = [bg.ndata.pop("atomic_number").to(device), bg.ndata.pop("chirality_type").to(device)]
    efeats = [bg.edata.pop("bond_type").to(device), bg.edata.pop("bond_direction_type").to(device)]
    with torch.no_grad():
        node_repr = model(bg, nfeats, efeats)
    return readout(bg, node_repr).detach().cpu().numpy()[0]


def graph_construction_and_featurization(smiles):
    """
    Constructs graphs from SMILES and featurizes them.

    Args:
        smiles (list of str): SMILES of molecules, for embedding computation.

    Returns:
        graphs (list of DGLGraph): List of graphs constructed and featurized.
        success (list of bool): Indicators for whether the SMILES string can be
            parsed by RDKit.
    """
    graphs = []
    success = []
    for smi in tqdm(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(
                mol,
                add_self_loop=True,
                node_featurizer=PretrainAtomFeaturizer(),
                edge_featurizer=PretrainBondFeaturizer(),
                canonical_atom_order=False,
            )
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)

    return graphs, success
