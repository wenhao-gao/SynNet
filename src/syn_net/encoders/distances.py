import numpy as np
from syn_net.encoders.fingerprints import mol_fp

def cosine_distance(v1, v2, eps=1e-15):
    """Computes the cosine similarity between two vectors.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        eps (float, optional): Small value, for numerical stability. Defaults
            to 1e-15.

    Returns:
        float: The cosine similarity.
    """
    return (1 - np.dot(v1, v2)
            / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2) + eps))

def ce_distance(y, y_pred, eps=1e-15):
    """Computes the cross-entropy between two vectors.

    Args:
        y (np.ndarray): First vector.
        y_pred (np.ndarray): Second vector.
        eps (float, optional): Small value, for numerical stability. Defaults
            to 1e-15.

    Returns:
        float: The cross-entropy.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - np.sum((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))


def _tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray):
    """
    Returns the Tanimoto similarity between two molecular fingerprints.

    Args:
        fp1 (np.ndarray): Molecular fingerprint 1.
        fp2 (np.ndarray): Molecular fingerprint 2.

    Returns:
        np.float: Tanimoto similarity.
    """
    return np.sum(fp1 * fp2) / (np.sum(fp1) + np.sum(fp2) - np.sum(fp1 * fp2))

def tanimoto_similarity(target_fp: np.ndarray, smis: list[str]):
    """
    Returns the Tanimoto similarities between a target fingerprint and molecules
    in an input list of SMILES.

    Args:
        target_fp (np.ndarray): Contains the reference (target) fingerprint.
        smis (list of str): Contains SMILES to compute similarity to.

    Returns:
        list of np.ndarray: Contains Tanimoto similarities.
    """
    fps = [mol_fp(smi, 2, 4096) for smi in smis]
    return [_tanimoto_similarity(target_fp, fp) for fp in fps]

