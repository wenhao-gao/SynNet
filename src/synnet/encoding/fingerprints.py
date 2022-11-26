from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def fp_embedding(smi: str, _radius: Optional[int] = 2, _nBits: Optional[int] = 4096) -> np.ndarray:
    """
    General function for building variable-size & -radius Morgan fingerprints.

    Args:
        smi (str): The SMILES to encode.
        _radius (int, optional): Morgan fingerprint radius. Defaults to 2.
        _nBits (int, optional): Morgan fingerprint length. Defaults to 4096.

    Returns:
        np.ndarray: A Morgan fingerprint generated using the specified parameters.
    """
    if smi is None:
        return np.zeros(_nBits).reshape((-1,))
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape((-1,))


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
