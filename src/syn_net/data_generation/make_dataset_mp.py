"""
This file generates synthetic tree data in a multi-thread fashion.

Usage:
    python make_dataset_mp.py
"""
import multiprocessing as mp

import numpy as np
from pathlib import Path
from syn_net.data_generation.make_dataset import synthetic_tree_generator
from syn_net.utils.data_utils import ReactionSet, SyntheticTreeSet
from syn_net.config import BUILDING_BLOCKS_RAW_DIR, DATA_PREPROCESS_DIR, MAX_PROCESSES
from syn_net.data_generation.preprocessing import BuildingBlockFileHandler
import logging

logger = logging.getLogger(__name__)


def func(_x):
    np.random.seed(_x) # dummy input to generate "unique" seed
    tree, action = synthetic_tree_generator(building_blocks, rxns)
    return tree, action


if __name__ == '__main__':

    reaction_template_id = "hb"  # "pis" or "hb"
    building_blocks_id = "enamine_us-2021-smiles"
    NUM_TREES = 600_000

    # Load building blocks
    building_blocks_file = Path(BUILDING_BLOCKS_RAW_DIR) / f"{building_blocks_id}.csv.gz"
    building_blocks = BuildingBlockFileHandler.load(building_blocks_file)

    # Load genearted reactions (matched reactions <=> building blocks)
    reactions_dir = Path(DATA_PREPROCESS_DIR)
    reactions_file =  f"reaction-sets_{reaction_template_id}_{building_blocks_id}.json.gz"
    r_set = ReactionSet().load(reactions_dir / reactions_file)
    rxns = r_set.rxns

    # Generate synthetic trees
    with mp.Pool(processes=MAX_PROCESSES) as pool:
        results = pool.map(func, np.arange(NUM_TREES).tolist())

    # Filter out trees that were completed with action="end"
    trees = [r[0] for r in results if r[1] == 3]
    actions = [r[1] for r in results]

    num_finish = actions.count(3)
    num_error = actions.count(-1)
    num_unfinish = NUM_TREES - num_finish - num_error

    logging.info(f"Total trial {NUM_TREES}")
    logging.info(f"Number of finished trees: {num_finish}")
    logging.info(f"Number of of unfinished tree: {num_unfinish}")
    logging.info(f"Number of error processes: {num_error}")

    # Save to local disk
    tree_set = SyntheticTreeSet(trees)
    outfile = f"synthetic-trees_{reaction_template_id}-{building_blocks_id}.json.gz"
    file = Path(DATA_PREPROCESS_DIR) / outfile
    tree_set.save(file)
