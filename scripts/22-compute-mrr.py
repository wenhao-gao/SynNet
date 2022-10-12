"""Compute the mean reciprocal ranking for reactant 1
selection using the different distance metrics in the k-NN search.
"""
import json
import logging

import numpy as np
from tqdm import tqdm

from synnet.config import MAX_PROCESSES
from synnet.encoding.distances import ce_distance, cosine_distance
from synnet.models.common import xy_to_dataloader
from synnet.models.mlp import load_mlp_from_ckpt
from synnet.MolEmbedder import MolEmbedder

logger = logging.getLogger(__name__)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-file", type=str, help="Checkpoint to load trained reactant 1 network."
    )
    parser.add_argument(
        "--embeddings-file", type=str, help="Pre-computed molecular embeddings for kNN search."
    )
    parser.add_argument("--X-data-file", type=str, help="Featurized X data for network.")
    parser.add_argument("--y-data-file", type=str, help="Featurized y data for network.")
    parser.add_argument(
        "--nbits", type=int, default=4096, help="Number of Bits for Morgan fingerprint."
    )
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "chebyshev", "cross_entropy", "cosine"],
        help="Distance function for `BallTree`.",
    )
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Init BallTree for kNN-search
    if args.distance == "cross_entropy":
        metric = ce_distance
    elif args.distance == "cosine":
        metric = cosine_distance
    else:
        metric = args.distance

    # Recall default: Morgan fingerprint with radius=2, nbits=256
    mol_embedder = MolEmbedder().load_precomputed(args.embeddings_file)
    mol_embedder.init_balltree(metric=metric)
    n, d = mol_embedder.embeddings.shape

    # Load data
    dataloader = xy_to_dataloader(
        X_file=args.X_data_file,
        y_file=args.y_data_file,
        n=None if not args.debug else 128,
        batch_size=args.batch_size,
        num_workers=args.ncpu,
        shuffle=False,
    )

    # Load MLP
    rt1_net = load_mlp_from_ckpt(args.ckpt_file)
    rt1_net.to(args.device)

    ranks = []
    for X, y in tqdm(dataloader):
        X, y = X.to(args.device), y.to(args.device)
        y_hat = rt1_net(X)  # (batch_size,nbits)

        ind_true = mol_embedder.kdtree.query(y.detach().cpu().numpy(), k=1, return_distance=False)
        ind = mol_embedder.kdtree.query(y_hat.detach().cpu().numpy(), k=n, return_distance=False)

        irows, icols = np.nonzero(ind == ind_true)  # irows = range(batch_size), icols = ranks
        ranks.append(icols)

    ranks = np.asarray(ranks, dtype=int).flatten()  # (nSamples,)
    rrs = 1 / (ranks + 1)  # +1 for offset 0-based indexing

    # np.save("ranks_" + metric + ".npy", ranks)  # TODO: do not hard code

    print(f"Result using metric: {metric}")
    print(f"The mean reciprocal ranking is: {rrs.mean():.3f}")
    TOP_N_RANKS = (1, 3, 5, 10, 15, 30)
    for i in TOP_N_RANKS:
        n_recovered = sum(ranks < i)
        n = len(ranks)
        print(f"The Top-{i:<2d} recovery rate is: {n_recovered/n:.3f} ({n_recovered}/{n})")
