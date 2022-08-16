"""
Filters the synthetic trees by the QEDs of the root molecules.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from tdc import Oracle
from tqdm import tqdm

from syn_net.config import DATA_PREPROCESS_DIR
from syn_net.utils.data_utils import SyntheticTree, SyntheticTreeSet

DATA_DIR = "pool001/whgao/data/synth_net"
SYNTHETIC_TREES_FILE = "abc-st_data.json.gz"

def _is_valid_mol(mol: Chem.rdchem.Mol):
    return mol is not None

if __name__ == '__main__':
    reaction_template_id = "hb"  # "pis" or "hb" 
    building_blocks_id = "enamine_us-2021-smiles"
    qed = Oracle(name='qed')

    # Load generated synthetic trees
    file =  Path(DATA_PREPROCESS_DIR) / f"synthetic-trees_{reaction_template_id}-{building_blocks_id}.json.gz"
    st_set = SyntheticTreeSet()
    st_set.load(file)
    synthetic_trees = st_set.sts
    print(f'Finish reading, in total {len(synthetic_trees)} synthetic trees.')

    # Filter synthetic trees 
    #  .. based on validity of root molecule
    #  .. based on drug-like quality
    filtered_data: list[SyntheticTree] = []
    original_qed: list[float] = []
    qeds: list[float] = []
    generated_smiles: list[str] = []

    threshold = 0.5

    for t in tqdm(synthetic_trees):
        try:
            smiles = t.root.smiles
            mol = Chem.MolFromSmiles(smiles)
            if not _is_valid_mol(mol):
                continue
            if smiles in generated_smiles:
                continue

            qed_value = qed(smiles)
            original_qed.append(qed_value)

            # filter the trees based on their QEDs
            if qed_value > threshold or np.random.random() < (qed_value/threshold):
                generated_smiles.append(smiles)
                filtered_data.append(t)
                qeds.append(qed_value)
            
        except Exception as e:
            print(e)

    print(f'Finish sampling, remaining {len(filtered_data)} synthetic trees.')

    # Save to local disk
    st_set = SyntheticTreeSet(filtered_data)
    file =  Path(DATA_PREPROCESS_DIR) / f"synthetic-trees_{reaction_template_id}-{building_blocks_id}-filtered.json.gz"
    st_set.save(file)

    df = pd.DataFrame({'SMILES': generated_smiles, 'qed': qeds})
    file = Path(DATA_PREPROCESS_DIR) / f"filtered-smiles_{reaction_template_id}-{building_blocks_id}-filtered.csv.gz"
    df.to_csv(file, compression='gzip', index=False)

    print('Finish!')
