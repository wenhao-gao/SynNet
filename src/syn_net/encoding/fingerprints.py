import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

## Morgan fingerprints
def mol_fp(smi, _radius=2, _nBits=4096):
    """
    Computes the Morgan fingerprint for the input SMILES.

    Args:
        smi (str): SMILES for molecule to compute fingerprint for.
        _radius (int, optional): Fingerprint radius to use. Defaults to 2.
        _nBits (int, optional): Length of fingerprint. Defaults to 1024.

    Returns:
        features (np.ndarray): For valid SMILES, this is the fingerprint.
            Otherwise, if the input SMILES is bad, this will be a zero vector.
    """
    if smi is None:
        return np.zeros(_nBits)
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, _radius, _nBits)
        return np.array(features_vec) # TODO: much slower compared to `DataStructs.ConvertToNumpyArray` (20x?) so deprecates

def fp_embedding(smi, _radius=2, _nBits=4096):
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