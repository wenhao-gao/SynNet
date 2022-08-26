"""
Prepares the training, testing, and validation data by reading in the states
and steps for the reaction data and re-writing it as separate one-hot encoded
Action, Reactant 1, Reactant 2, and Reaction files.
"""
import logging
from pathlib import Path

from syn_net.config import DATA_FEATURIZED_DIR
from syn_net.utils.prep_utils import prep_data

logger = logging.getLogger(__file__)

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
    parser.add_argument("-rxn", "--rxn_template", type=str, default='hb', choices=["hb","pis"],
                        help="Choose from ['hb', 'pis']")

    args = parser.parse_args()
    reaction_template_id = args.rxn_template
    embedding = args.targetembedding
    output_emb = args.outputembedding

    main_dir = Path(DATA_FEATURIZED_DIR) / f'{reaction_template_id}_{embedding}_{args.radius}_{args.nbits}_{args.outputembedding}/' # must match with dir in `st2steps.py`
    if reaction_template_id == 'hb':
        num_rxn = 91
    elif reaction_template_id == 'pis':
        num_rxn = 4700

    # Get dimension of output embedding
    OUTPUT_EMBEDDINGS = {
        "gin": 300,
        "fp_4096": 4096,
        "fp_256": 256,
        "rdkit2d": 200,
    }
    out_dim = OUTPUT_EMBEDDINGS[output_emb]

    logger.info("Start splitting data.")
    # Split datasets for each MLP
    prep_data(main_dir, num_rxn, out_dim)

    logger.info("Successfully splitted data.")
