"""
Splits a synthetic tree into states and steps.
"""
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
from syn_net.utils.data_utils import SyntheticTreeSet
from syn_net.utils.prep_utils import organize

import logging
logger = logging.getLogger(__file__)

from syn_net.config import DATA_PREPARED_DIR, DATA_FEATURIZED_DIR

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--targetembedding", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-o", "--outputembedding", type=str, default='fp_256',
                        help="Choose from ['fp_4096', 'fp_256', 'gin', 'rdkit2d']")
    parser.add_argument("-r", "--radius", type=int, default=2,
                        help="Radius for Morgan Fingerprint")
    parser.add_argument("-b", "--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan Fingerprint")
    parser.add_argument("-d", "--datasettype", type=str, choices=["train","valid","test"],
                        help="Choose from ['train', 'valid', 'test']")
    parser.add_argument("-rxn", "--rxn_template", type=str, default='hb', choices=["hb","pis"],
                        help="Choose from ['hb', 'pis']")
    args = parser.parse_args()
    logger.info(vars(args))

    # Parse & set inputs
    reaction_template_id = args.rxn_template
    building_blocks_id = "enamine_us-2021-smiles"
    dataset_type = args.datasettype
    embedding = args.targetembedding
    assert dataset_type is not None, "Must specify which dataset to use."

    # Load synthetic trees subset {train,valid,test}
    file = f'{DATA_PREPARED_DIR}/synthetic-trees-{dataset_type}.json.gz'
    st_set = SyntheticTreeSet()
    st_set.load(file)
    logger.info("Number of synthetic trees: {len(st_set.sts}")
    data: list = st_set.sts
    del st_set
    
    # Set output directory
    save_dir = Path(DATA_FEATURIZED_DIR) / f'{reaction_template_id}_{embedding}_{args.radius}_{args.nbits}_{args.outputembedding}/'
    Path(save_dir).mkdir(parents=1,exist_ok=1)

    # Start splitting synthetic trees in states and steps
    states = []
    steps = []

    for st in tqdm(data):
        try:
            state, step = organize(st, target_embedding=embedding,
            radius=args.radius,
            nBits=args.nbits,
            output_embedding=args.outputembedding)
        except Exception as e:
            logger.exception(exc_info=e)
            continue
        states.append(state)
        steps.append(step)


    # Finally, save. 
    logger.info(f"Saving to {save_dir}")
    states = sparse.vstack(states)
    steps = sparse.vstack(steps)
    sparse.save_npz(save_dir / f"states_{dataset_type}.npz", states)
    sparse.save_npz(save_dir / f"steps_{dataset_type}.npz", steps)

    logger.info("Save successful.")

