"""
This file contains various utils for data preparation and preprocessing.
"""
from typing import Iterator, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from syn_net.utils.data_utils import Reaction, SyntheticTree
from syn_net.utils.predict_utils import (can_react, get_action_mask,
                                         get_reaction_mask, )
from syn_net.encoding.fingerprints import mol_fp

from pathlib import Path
from rdkit import Chem
import logging
logger = logging.getLogger(__name__)

def rdkit2d_embedding(smi):
    """
    Computes an embedding using RDKit 2D descriptors.

    Args:
        smi (str): SMILES string.

    Returns:
        np.ndarray: A molecular embedding corresponding to the input molecule.
    """
    from tdc.chem_utils import MolConvert
    if smi is None:
        return np.zeros(200).reshape((-1, ))
    else:
        # define the RDKit 2D descriptor
        rdkit2d = MolConvert(src = 'SMILES', dst = 'RDKit2D')
        return rdkit2d(smi).reshape(-1, )

import functools
@functools.lru_cache(maxsize=1)
def _fetch_gin_pretrained_model(model_name: str):
    from dgllife.model import load_pretrained
    """Get a GIN pretrained model to use for creating molecular embeddings"""
    device     = 'cpu'
    model      = load_pretrained(model_name).to(device)
    model.eval()
    return model


def split_data_into_Xy(
    dataset_type: str,
    steps_file: str,
    states_file: str,
    output_dir: Path,
    num_rxn: int,
    out_dim: int,
    ) -> None:
    """Split the featurized data into X,y-chunks for the {act,rt1,rxn,rt2}-networks.

    Args:
        num_rxn (int): Number of reactions in the dataset.
        out_dim (int): Size of the output feature vectors (used in kNN-search for rt1,rt2)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True,parents=True)

    # Load data # TODO: separate functionality?
    states = sparse.load_npz(states_file)
    steps = sparse.load_npz(steps_file)

    # Extract data for each network...

    # ... action data
    # X: [z_state]
    # y: [action id] (int)
    X = states
    y = steps[:, 0]
    sparse.save_npz(output_dir / f'X_act_{dataset_type}.npz', X)
    sparse.save_npz(output_dir / f'y_act_{dataset_type}.npz', y)
    logger.info(f'  saved data for "Action" to {output_dir}')

    # Delete all data where tree was ended (i.e. tree expansion did not trigger reaction)
    # TODO: Look into simpler slicing with boolean indices, perhabs consider CSR for row slicing
    states = sparse.csc_matrix(states.A[(steps[:, 0].A != 3).reshape(-1, )])
    steps  = sparse.csc_matrix(steps.A[(steps[:, 0].A != 3).reshape(-1, )])

    # ... reaction data
    # X: [state, z_reactant_1]
    # y: [reaction_id] (int)
    X = sparse.hstack([states, steps[:, (2 * out_dim + 2):]])
    y = steps[:, out_dim + 1]
    sparse.save_npz(output_dir / f'X_rxn_{dataset_type}.npz', X)
    sparse.save_npz(output_dir / f'y_rxn_{dataset_type}.npz', y)
    logger.info(f'  saved data for "Reaction" to {output_dir}')

    states = sparse.csc_matrix(states.A[(steps[:, 0].A != 2).reshape(-1, )])
    steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 2).reshape(-1, )])

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit([[i] for i in range(num_rxn)])

    # ... reactant 2 data
    # X: [z_state, z_reactant_1, reaction_id]
    # y: [z'_reactant_2]
    X = sparse.hstack(
        [states,
            steps[:, (2 * out_dim + 2):],
            sparse.csc_matrix(enc.transform(steps[:, out_dim+1].A.reshape((-1, 1))).toarray())]
    )
    y = steps[:, (out_dim+2): (2 * out_dim + 2)]
    sparse.save_npz(output_dir / f'X_rt2_{dataset_type}.npz', X)
    sparse.save_npz(output_dir / f'y_rt2_{dataset_type}.npz', y)
    logger.info(f'  saved data for "Reactant 2" to {output_dir}')

    states = sparse.csc_matrix(states.A[(steps[:, 0].A != 1).reshape(-1, )])
    steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 1).reshape(-1, )])

    # ... reactant 1 data
    # X: [z_state]
    # y: [z'_reactant_1]
    X = states
    y = steps[:, 1: (out_dim+1)]
    sparse.save_npz(output_dir / f'X_rt1_{dataset_type}.npz', X)
    sparse.save_npz(output_dir / f'y_rt1_{dataset_type}.npz', y)
    logger.info(f'  saved data for "Reactant 1" to {output_dir}')

    return None

class Sdf2SmilesExtractor:
    """Helper class for data generation."""

    def __init__(self) -> None:
        self.smiles: Iterator[str]

    def from_sdf(self, file: Union[str, Path]):
        """Extract chemicals as SMILES from `*.sdf` file.

        See also:
            https://www.rdkit.org/docs/GettingStartedInPython.html#reading-sets-of-molecules
        """
        file = str(Path(file).resolve())
        suppl = Chem.SDMolSupplier(file)
        self.smiles = (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in suppl)
        logger.info(f"Read data from {file}")

        return self

    def _to_csv_gz(self, file: Path) -> None:
        import gzip

        with gzip.open(file, "wt") as f:
            f.writelines("SMILES\n")
            f.writelines((s + "\n" for s in self.smiles))

    def _to_txt(self, file: Path) -> None:
        with open(file, "wt") as f:
            f.writelines("SMILES\n")
            f.writelines((s + "\n" for s in self.smiles))

    def to_file(self, file: Union[str, Path]) -> None:

        if Path(file).suffixes == [".csv", ".gz"]:
            self._to_csv_gz(file)
        else:
            self._to_txt(file)
        logger.info(f"Saved data to {file}")