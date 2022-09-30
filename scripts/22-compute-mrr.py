"""
This function is used to compute the mean reciprocal ranking for reactant 1
selection using the different distance metrics in the k-NN search.
"""
from syn_net.models.mlp import MLP, load_array
from scipy import sparse
import numpy as np
from sklearn.neighbors import BallTree
import torch
from syn_net.encoding.distances import cosine_distance, ce_distance

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-file", type=str,help="Checkpoint to load trained reactant 1 network.")
    parser.add_argument("--embeddings-file", type=str,help="Pre-computed molecular embeddings for kNN search.")
    parser.add_argument("--X-data-file", type=str, help="Featurized X data for network.")
    parser.add_argument("--y-data-file", type=str, help="Featurized y data for network.")
    parser.add_argument("--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="")
    parser.add_argument("--distance", type=str, default="euclidean",
                        choices=['euclidean', 'manhattan', 'chebyshev', 'cross_entropy', 'cosine'],help="Distance function for `BallTree`.")
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()


    bb_emb_fp_256 = np.load(args.embeddings_file)
    n, d = bb_emb_fp_256.shape

    metric = args.distance
    if metric == 'cross_entropy':
        metric = ce_distance
    elif metric == 'cosine':
        metric = cosine_distance

    kdtree_fp_256 = BallTree(bb_emb_fp_256, metric=metric)

    path_to_rt1 = args.ckpt_file
    batch_size = args.batch_size
    ncpu = args.ncpu

    X = sparse.load_npz(args.X_data_file)
    y = sparse.load_npz(args.y_data_file)
    X = torch.Tensor(X.A)
    y = torch.Tensor(y.A)
    _idx = np.random.choice(list(range(X.shape[0])), size=int(X.shape[0]/10), replace=False)
    test_data_iter = load_array((X[_idx], y[_idx]), batch_size, ncpu=ncpu, is_train=False)
    data_iter = test_data_iter

    rt1_net = MLP.load_from_checkpoint(path_to_rt1,
                    input_dim=int(3 * args.nbits),
                    output_dim=d,
                    hidden_dim=1200,
                    num_layers=5,
                    dropout=0.5,
                    num_dropout_layers=1,
                    task='regression',
                    loss='mse',
                    valid_loss='mse',
                    optimizer='adam',
                    learning_rate=1e-4,
                    ncpu=ncpu)
    rt1_net.eval()
    rt1_net.to(args.device)



    ranks = []
    for X, y in data_iter:
        X, y = X.to(args.device), y.to(args.device)
        y_hat = rt1_net(X)
        dist_true, ind_true = kdtree_fp_256.query(y.detach().cpu().numpy(), k=1)
        dist, ind = kdtree_fp_256.query(y_hat.detach().cpu().numpy(), k=n)
        ranks = ranks + [np.where(ind[i] == ind_true[i])[0][0] for i in range(len(ind_true))]

    ranks = np.array(ranks)
    rrs = 1 / (ranks + 1)

    np.save('ranks_' + metric + '.npy', ranks) # TODO: do not hard code

    print(f"Result using metric: {metric}")
    print(f"The mean reciprocal ranking is: {rrs.mean():.3f}")
    print(f"The Top-1 recovery rate is: {sum(ranks < 1) / len(ranks) :.3f}, {sum(ranks < 1)} / {len(ranks)}")
    print(f"The Top-3 recovery rate is: {sum(ranks < 3) / len(ranks) :.3f}, {sum(ranks < 3)} / {len(ranks)}")
    print(f"The Top-5 recovery rate is: {sum(ranks < 5) / len(ranks) :.3f}, {sum(ranks < 5)} / {len(ranks)}")
    print(f"The Top-10 recovery rate is: {sum(ranks < 10) / len(ranks) :.3f}, {sum(ranks < 10)} / {len(ranks)}")
    print(f"The Top-15 recovery rate is: {sum(ranks < 15) / len(ranks) :.3f}, {sum(ranks < 15)} / {len(ranks)}")
    print(f"The Top-30 recovery rate is: {sum(ranks < 30) / len(ranks) :.3f}, {sum(ranks < 30)} / {len(ranks)}")
    print()
