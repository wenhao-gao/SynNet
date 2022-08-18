"""
Reads synthetic tree data and splits it into training, validation and testing sets.
"""
from syn_net.utils.data_utils import SyntheticTreeSet
from pathlib import Path
from syn_net.config import DATA_PREPROCESS_DIR, DATA_PREPARED_DIR

if __name__ == "__main__":
    reaction_template_id = "hb"  # "pis" or "hb" 
    building_blocks_id = "enamine_us-2021-smiles"

    # Load filtered synthetic trees
    st_set = SyntheticTreeSet()
    file =  Path(DATA_PREPROCESS_DIR) / f"synthetic-trees_{reaction_template_id}-{building_blocks_id}-filtered.json.gz"
    print(f'Reading data from {file}')
    st_set.load(file)
    data = st_set.sts
    del st_set
    num_total = len(data)
    print(f"There are {len(data)} synthetic trees.")

    # Split data
    SPLIT_RATIO = [0.6, 0.2, 0.2]

    num_train = int(SPLIT_RATIO[0] * num_total)
    num_valid = int(SPLIT_RATIO[1] * num_total)
    num_test = num_total - num_train - num_valid

    data_train = data[:num_train]
    data_valid = data[num_train: num_train + num_valid]
    data_test = data[num_train + num_valid: ]

    # Save to local disk

    print("Saving training dataset: ", len(data_train))
    trees = SyntheticTreeSet(data_train)
    trees.save(f'{DATA_PREPARED_DIR}/synthetic-trees-train.json.gz')

    print("Saving validation dataset: ", len(data_valid))
    trees = SyntheticTreeSet(data_valid)
    trees.save(f'{DATA_PREPARED_DIR}/synthetic-trees-valid.json.gz')

    print("Saving testing dataset: ", len(data_test))
    trees = SyntheticTreeSet(data_test)
    trees.save(f'{DATA_PREPARED_DIR}/synthetic-trees-test.json.gz')

    print("Finish!")
