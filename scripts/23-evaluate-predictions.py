"""Evaluate a batch of predictions on different metrics.
The predictions are generated in `20-predict-targets.py`.
"""
import json
import logging

import numpy as np
import pandas as pd
from tdc import Evaluator

from synnet.config import MAX_PROCESSES

logger = logging.getLogger(__name__)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        type=str,
        help="Dataframe with target- and prediction smiles and similarities (*.csv.gz).",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Keep track of successfully and unsuccessfully recovered molecules in 2 df's
    # NOTE: column names must match input dataframe...
    recovered = pd.DataFrame({"targets": [], "decoded": [], "similarity": []})
    unrecovered = pd.DataFrame({"targets": [], "decoded": [], "similarity": []})

    # load each file containing the predictions
    similarity = []
    n_recovered = 0
    n_unrecovered = 0
    n_total = 0
    files = [args.input_file]  # TODO: not sure why the loop but let's keep it for now
    for file in files:
        print(f"Evaluating file: {file}")

        result_df = pd.read_csv(file)
        n_total += len(result_df["decoded"])

        # Split smiles, discard NaNs
        is_recovered = result_df["similarity"] == 1.0
        unrecovered = pd.concat([unrecovered, result_df[~is_recovered].dropna()])
        recovered = pd.concat([recovered, result_df[is_recovered].dropna()])

        n_recovered += len(recovered)
        n_unrecovered += len(unrecovered)
        similarity += unrecovered["similarity"].tolist()

    # Print general info
    print(f"N total {n_total}")
    print(f"N recovered {n_recovered} ({n_recovered/n_total:.2f})")
    print(f"N unrecovered {n_unrecovered} ({n_recovered/n_total:.2f})")

    n_finished = n_recovered + n_unrecovered
    n_unfinished = n_total - n_finished
    print(f"N finished tree {n_finished} ({n_finished/n_total:.2f})")
    print(f"N unfinished trees (NaN) {n_unfinished} ({n_unfinished/n_total:.2f})")
    print(f"Average similarity (unrecovered only) {np.mean(similarity)}")

    # Evaluate on TDC evaluators
    for metric in "KL_divergence FCD_Distance Novelty Validity Uniqueness".split():
        evaluator = Evaluator(name=metric)
        try:
            score_recovered = evaluator(recovered["targets"], recovered["decoded"])
            score_unrecovered = evaluator(unrecovered["targets"], unrecovered["decoded"])
        except TypeError:
            # Some evaluators only take 1 input args, try that.
            score_recovered = evaluator(recovered["decoded"])
            score_unrecovered = evaluator(unrecovered["decoded"])
        except Exception as e:
            logger.error(f"{e.__class__.__name__}: {str(e)}")
            logger.error(e)
            score_recovered, score_unrecovered = np.nan, np.nan

        print(f"Evaluation metric for {evaluator.name}:")
        print(f"    Recovered score: {score_recovered:.2f}")
        print(f"  Unrecovered score: {score_unrecovered:.2f}")
