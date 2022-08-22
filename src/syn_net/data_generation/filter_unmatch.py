"""
Filters out purchasable building blocks which don't match a single template.
"""
from syn_net.utils.data_utils import *
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from syn_net.data_generation.process_rxn_mp import _load_building_blocks # TODO: refactor
from syn_net.config import BUILDING_BLOCKS_RAW_DIR, DATA_PREPROCESS_DIR
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    reaction_template_id = "hb"  # "pis" or "hb" 
    building_blocks_id = "enamine_us-2021-smiles"

    # Load building blocks
    building_blocks_file = Path(BUILDING_BLOCKS_RAW_DIR) / f"{building_blocks_id}.csv.gz"
    building_blocks = _load_building_blocks(building_blocks_file)


    # Load genearted reactions (matched reactions <=> building blocks)
    reactions_dir = Path(DATA_PREPROCESS_DIR)
    reactions_file =  f"reaction-sets_{reaction_template_id}_{building_blocks_id}.json.gz"
    r_set = ReactionSet().load(reactions_dir / reactions_file)

    # Identify all used building blocks (via union of sets)
    matched_bblocks = set()
    for r in tqdm(r_set.rxns):
        for reactants in r.available_reactants:
            matched_bblocks = matched_bblocks.union(set(reactants))


    logger.info(f'Total number of building blocks {len(building_blocks):d}')
    logger.info(f'Matched number of building blocks {len(matched_bblocks):d}')
    logger.info(f"{len(matched_bblocks)/len(building_blocks):.2%} of building blocks are applicable for the reaction template set '{reaction_template_id}'.")

    # Save to local disk
    df = pd.DataFrame({'SMILES': list(matched_bblocks)})
    outfile = f"{reaction_template_id}-{building_blocks_id}-matched.csv.gz"
    file = Path(DATA_PREPROCESS_DIR) / outfile
    df.to_csv(file, compression='gzip')
