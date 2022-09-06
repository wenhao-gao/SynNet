"""
This file processes a set of reaction templates and finds applicable
reactants from a list of purchasable building blocks.

Usage:
    python process_rxn.py
"""
import multiprocessing as mp
from functools import partial
from pathlib import Path
from time import time

# Silence RDKit loggers (https://github.com/rdkit/rdkit/issues/2683)
from rdkit import RDLogger

from syn_net.utils.data_utils import Reaction, ReactionSet

RDLogger.DisableLog("rdApp.*")


import pandas as pd


def _load_building_blocks(file: Path) -> list[str]:
    return pd.read_csv(file)["SMILES"].to_list()


def _match_building_blocks_to_rxn(building_blocks: list[str], _rxn: Reaction):
    _rxn.set_available_reactants(building_blocks)
    return _rxn


from syn_net.config import (BUILDING_BLOCKS_RAW_DIR, DATA_PREPROCESS_DIR,
                            REACTION_TEMPLATE_DIR)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-blocks-file", type=str, help="Input file with SMILES strings (First row `SMILES`, then one per line).")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    reaction_template_id = "hb"  # "pis" or "hb"
    building_blocks_id = "enamine_us-2021-smiles"

    # Load building blocks
    building_blocks_file = Path(BUILDING_BLOCKS_RAW_DIR) / f"{building_blocks_id}.csv.gz"
    building_blocks = _load_building_blocks(building_blocks_file)

    # Load reaction templates and parse
    path_to__rxntemplates = Path(REACTION_TEMPLATE_DIR) / f"{reaction_template_id}.txt"
    _rxntemplates = []
    for line in open(path_to__rxntemplates, "rt"):
        template = line.strip()
        rxn = Reaction(template)
        _rxntemplates.append(rxn)

    # Filter building blocks on each reaction
    t = time()
    func = partial(_match_building_blocks_to_rxn, building_blocks)
    with mp.Pool(processes=64) as pool:
        rxns = pool.map(func, _rxntemplates)
    print("Time: ", time() - t, "s")

    # Save data to local disk
    r = ReactionSet(rxns)
    out_dir = Path(DATA_PREPROCESS_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"reaction-sets_{reaction_template_id}_{building_blocks_id}.json.gz"
    r.save(out_file)
