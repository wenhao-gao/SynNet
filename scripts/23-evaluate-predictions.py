"""Evaluate a batch of predictions on different metrics.
The predictions are generated in `20-predict-targets.py`.
"""
from tdc import Evaluator
import pandas as pd
import numpy as np

kl_divergence = Evaluator(name = 'KL_Divergence')
fcd_distance = Evaluator(name = 'FCD_Distance')
novelty = Evaluator(name = 'Novelty')
validity = Evaluator(name = 'Validity')
uniqueness = Evaluator(name = 'Uniqueness')

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file", type=str, help="Dataframe with target- and prediction smiles and similarities (*.csv.gz)."
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    files = [args.file] # TODO: not sure why the loop but let's keep it for now

    # Keep track of successfully and unsuccessfully recovered molecules in 2 df's
    recovered = pd.DataFrame({'query SMILES': [], 'decode SMILES': [], 'similarity':[]})
    unrecovered = pd.DataFrame({'query SMILES': [], 'decode SMILES': [], 'similarity':[]})

    # load each file containing the predictions
    similarity = []
    n_recovered = 0
    n_unrecovered = 0
    n_total = 0
    for file in files:
        print(f'File currently being evaluated: {file}')

        result_df = pd.read_csv(file)
        n_total += len(result_df['decode SMILES'])

        # Split smiles, discard NaNs
        is_recovered = result_df['similarity'] == 1.0
        unrecovered = pd.concat([unrecovered, result_df[~is_recovered].dropna()])
        recovered   = pd.concat([recovered, result_df[is_recovered].dropna()])

        n_recovered += len(recovered)
        n_unrecovered += len(unrecovered)
        similarity += unrecovered['similarity'].tolist()

    # compute the following properties, using the TDC, for the succesfully recovered molecules
    recovered_novelty_all = novelty(
        recovered['query SMILES'].tolist(),
        recovered['decode SMILES'].tolist(),
        )
    recovered_validity_decode_all = validity(recovered['decode SMILES'].tolist())
    recovered_uniqueness_decode_all = uniqueness(recovered['decode SMILES'].tolist())
    recovered_fcd_distance_all = fcd_distance(
        recovered['query SMILES'].tolist(),
        recovered['decode SMILES'].tolist()
        )
    recovered_kl_divergence_all = kl_divergence(recovered['query SMILES'].tolist(), recovered['decode SMILES'].tolist())

    # compute the following properties, using the TDC, for the unrecovered molecules
    unrecovered_novelty_all = novelty(unrecovered['query SMILES'].tolist(), unrecovered['decode SMILES'].tolist())
    unrecovered_validity_decode_all = validity(unrecovered['decode SMILES'].tolist())
    unrecovered_uniqueness_decode_all = uniqueness(unrecovered['decode SMILES'].tolist())
    unrecovered_fcd_distance_all = fcd_distance(unrecovered['query SMILES'].tolist(), unrecovered['decode SMILES'].tolist())
    unrecovered_kl_divergence_all = kl_divergence(unrecovered['query SMILES'].tolist(), unrecovered['decode SMILES'].tolist())

    # Print info
    print(f'N total {n_total}')
    print(f'N recovered {n_recovered} ({n_recovered/n_total:.2f})')
    print(f'N unrecovered {n_unrecovered} ({n_recovered/n_total:.2f})')

    n_finished = n_recovered + n_unrecovered
    n_unfinished = n_total - n_finished
    print(f'N finished tree {n_finished} ({n_finished/n_total:.2f})')
    print(f'N unfinished trees (NaN) {n_unfinished} ({n_unfinished/n_total:.2f})')
    print(f'Average similarity (unrecovered only) {np.mean(similarity)}')

    print('Novelty, recovered:', recovered_novelty_all)
    print('Novelty, unrecovered:', unrecovered_novelty_all)

    print('Validity, decode molecules, recovered:', recovered_validity_decode_all)
    print('Validity, decode molecules, unrecovered:', unrecovered_validity_decode_all)

    print('Uniqueness, decode molecules, recovered:', recovered_uniqueness_decode_all)
    print('Uniqueness, decode molecules, unrecovered:', unrecovered_uniqueness_decode_all)

    print('FCD distance, recovered:', recovered_fcd_distance_all)
    print('FCD distance, unrecovered:', unrecovered_fcd_distance_all)

    print('KL divergence, recovered:', recovered_kl_divergence_all)
    print('KL divergence, unrecovered:', unrecovered_kl_divergence_all)
