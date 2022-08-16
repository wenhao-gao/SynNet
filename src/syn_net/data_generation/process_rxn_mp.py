"""
This file processes a set of reaction templates and finds applicable
reactants from a list of purchasable building blocks.

Usage:
    python process_rxn_mp.py
"""
import multiprocessing as mp
from time import time

from syn_net.utils.data_utils import Reaction, ReactionSet
import syn_net.data_generation._mp_process as process
from pathlib import Path
# Silence RDKit loggers (https://github.com/rdkit/rdkit/issues/2683)
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*') 

from syn_net.config import REACTION_TEMPLATE_DIR, DATA_DIR

if __name__ == '__main__':
    name = 'hb' # "pis" or "hb"

    # Load reaction templates and parse
    path_to_rxn_templates = f'{REACTION_TEMPLATE_DIR}/{name}.txt'
    rxn_templates = []
    for line in open(path_to_rxn_templates, 'rt'):
        template = line.strip() 
        rxn = Reaction(template)
        rxn_templates.append(rxn)

    # Filter building blocks on each reaction
    pool = mp.Pool(processes=64)
    t = time()
    rxns = pool.map(process.func, rxn_templates)
    print('Time: ', time() - t, 's')

    # Save data to local disk
    r = ReactionSet(rxns)
    out_dir = Path(DATA_DIR) / f"pre-process/"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"st_{name}.json.gz"
    r.save(out_file)
