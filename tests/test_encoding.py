import numpy as np
import pytest

from synnet.encoding.fingerprints import fp_embedding


@pytest.fixture
def valid_smiles() -> list[str]:
    return [
        "CC(C)(C)C(C(=O)O)n1cn[nH]c1=O",
        "C=CC1CN(C(=O)OC(C)(C)C)CCC1CCO",
        "COC(=O)c1coc(-c2cnn(C)c2)n1",
        "CC(C)(C)OC(=O)N1CCN(c2cc(N)cc(Br)c2)CC1",
        "CN(C)C(=O)N1CCCNCC1.C",
    ]


@pytest.fixture
def valid_smi(valid_smiles) -> str:
    return valid_smiles[0]


def test_default_fp_embedding_single(valid_smi):
    fp = fp_embedding(valid_smi)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (4096,)


def test_default_fp_embedding_multiple(valid_smiles):
    n = len(valid_smiles)
    fps = np.asarray([fp_embedding(smi) for smi in valid_smiles])
    assert fps.shape == (n, 4096)


def test_fp_on_invalid_smiles(smi=None):
    fp = fp_embedding(smi)
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (4096,)
    assert fp.sum() == 0


def test_some_good_some_none(valid_smiles):
    n = len(valid_smiles)
    m = 3
    fps = np.asarray([fp_embedding(smi) for smi in valid_smiles + [None] * m])
    assert fps.shape == (n + m, 4096)


@pytest.mark.parametrize("nbits", [2**p for p in range(8, 13)])
def test_fp_None_dim(nbits):
    smi = None
    fp = fp_embedding(smi, _radius=2, _nBits=nbits)
    assert fp.shape == (nbits,)


@pytest.mark.parametrize("nbits", [2**p for p in range(8, 13)])
def test_fp_smiles_dim(nbits, valid_smi):
    smi = valid_smi
    fp = fp_embedding(smi, _radius=2, _nBits=nbits)
    assert fp.shape == (nbits,)
