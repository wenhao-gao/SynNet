"""
Reactant2 network (for predicting 2nd reactant).
"""
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from syn_net.models.mlp import MLP, load_array
from scipy import sparse
from syn_net.config import DATA_FEATURIZED_DIR
from pathlib import Path

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--out_dim", type=int, default=256,
                        help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epoch", type=int, default=2000,
                        help="Maximum number of epoches.")
    args = parser.parse_args()

    # Helper to select validation func based on output dim
    VALIDATION_OPTS = {
        300: "nn_accuracy_gin",
        4096: "nn_accuracy_fp_4096",
        256: "nn_accuracy_fp_256",
        200: "nn_accuracy_rdkit2d",
    }
    validation_option = VALIDATION_OPTS[args.out_dim]

    id = f'{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_{validation_option[12:]}/'
    main_dir   = Path(DATA_FEATURIZED_DIR) / id
    batch_size = args.batch_size
    ncpu       = args.ncpu


    X = sparse.load_npz(main_dir / 'X_rt2_train.npz')
    y = sparse.load_npz(main_dir / 'y_rt2_train.npz')
    X = torch.Tensor(X.A)
    y = torch.Tensor(y.A)
    train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

    X    = sparse.load_npz(main_dir / 'X_rt2_valid.npz')
    y    = sparse.load_npz(main_dir / 'y_rt2_valid.npz')
    X    = torch.Tensor(X.A)
    y    = torch.Tensor(y.A)
    _idx = np.random.choice(list(range(X.shape[0])), size=int(X.shape[0]/10), replace=False)
    valid_data_iter = load_array((X[_idx], y[_idx]), batch_size, ncpu=ncpu, is_train=False)

    pl.seed_everything(0)
    INPUT_DIMS = {
        "fp": {
            "hb": int(4 * args.nbits + 91),
            "gin": int(4 * args.nbits + 4700),
        },
        "gin" : {
            "hb": int(3 * args.nbits + args.out_dim + 91),
            "gin": int(3 * args.nbits + args.out_dim + 4700),
        }
    } # somewhat constant...
    input_dims = INPUT_DIMS[args.featurize][args.rxn_template]

    mlp = MLP(input_dim=input_dims,
                output_dim=args.out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='regression',
                loss='mse',
                valid_loss=validation_option,
                optimizer='adam',
                learning_rate=1e-4,
                val_freq=10,
                ncpu=ncpu)

    tb_logger = pl_loggers.TensorBoardLogger(
        f'rt2_{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_{validation_option[12:]}_logs/'
    )

    trainer = pl.Trainer(gpus=[0], max_epochs=args.epoch, progress_bar_refresh_rate=20, logger=tb_logger)
    t       = time.time()
    trainer.fit(mlp, train_data_iter, valid_data_iter)
    print(time.time() - t, 's')

    print('Finish!')
