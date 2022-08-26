"""
Reaction network.
"""
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from scipy import sparse

from syn_net.config import CHECKPOINTS_DIR, DATA_FEATURIZED_DIR
from syn_net.models.common import VALIDATION_OPTS, get_args
from syn_net.models.mlp import MLP, load_array

if __name__ == '__main__':

    args = get_args()

    validation_option = VALIDATION_OPTS[args.out_dim]

    id = f'{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_{validation_option[12:]}/'
    main_dir   = Path(DATA_FEATURIZED_DIR) / id
    batch_size = args.batch_size
    ncpu       = args.ncpu

    X = sparse.load_npz(main_dir / 'X_rxn_train.npz')
    y = sparse.load_npz(main_dir / 'y_rxn_train.npz')
    X = torch.Tensor(X.A)
    y = torch.LongTensor(y.A.reshape(-1, ))
    train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

    X = sparse.load_npz(main_dir / 'X_rxn_valid.npz')
    y = sparse.load_npz(main_dir / 'y_rxn_valid.npz')
    X = torch.Tensor(X.A)
    y = torch.LongTensor(y.A.reshape(-1, ))
    valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

    pl.seed_everything(0)
    param_path  = Path(CHECKPOINTS_DIR) / f"{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_v{args.version}/"
    path_to_rxn = f'{param_path}rxn.ckpt'

    INPUT_DIMS = {
        "fp": {
            "hb": int(4 * args.nbits),
            "gin": int(4 * args.nbits),
        },
        "gin" : {
            "hb": int(3 * args.nbits + args.out_dim),
            "gin": int(3 * args.nbits + args.out_dim),
        }
    } # somewhat constant...
    input_dim = INPUT_DIMS[args.featurize][args.rxn_template]

    HIDDEN_DIMS = {
        "fp": {
            "hb": 3000,
            "gin": 4500,
        },
        "gin" : {
            "hb": 3000,
            "gin": 3000,
        }
    }
    hidden_dim = HIDDEN_DIMS[args.featurize][args.rxn_template]

    OUTPUT_DIMS = {
            "hb": 91,
            "gin": 4700,
    }
    output_dim = OUTPUT_DIMS[args.rxn_template]


    if not args.restart:
        mlp = MLP(input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=5,
                    dropout=0.5,
                    num_dropout_layers=1,
                    task='classification',
                    loss='cross_entropy',
                    valid_loss='accuracy',
                    optimizer='adam',
                    learning_rate=1e-4,
                    val_freq=10,
                    ncpu=ncpu,
                )
    else: # load from checkpt -> only for fp, not gin
        mlp = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )

    tb_logger = pl_loggers.TensorBoardLogger(f'rxn_{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_logs/')
    trainer   = pl.Trainer(gpus=[0], max_epochs=args.epoch, progress_bar_refresh_rate=20, logger=tb_logger)
    t         = time.time()

    trainer.fit(mlp, train_data_iter, valid_data_iter)

    print(time.time() - t, 's')

    print('Finish!')
