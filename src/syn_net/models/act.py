"""
Action network.
"""
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from syn_net.models.mlp import MLP, load_array
from scipy import sparse
from syn_net.config import DATA_FEATURIZED_DIR
from pathlib import Path
from syn_net.models.common import get_args

if __name__ == '__main__':

    args = get_args()

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

    X = sparse.load_npz(main_dir / 'X_act_train.npz')
    y = sparse.load_npz(main_dir / 'y_act_train.npz')
    X = torch.Tensor(X.A)
    y = torch.LongTensor(y.A.reshape(-1, ))
    train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

    X = sparse.load_npz(main_dir / 'X_act_valid.npz')
    y = sparse.load_npz(main_dir / 'y_act_valid.npz')
    X = torch.Tensor(X.A)
    y = torch.LongTensor(y.A.reshape(-1, ))
    valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

    pl.seed_everything(0)
    INPUT_DIMS = {
        "fp": int(3 * args.nbits),
        "gin" : int(2 * args.nbits + args.out_dim)
    } # somewhat constant...

    input_dims = INPUT_DIMS[args.featurize]

    mlp = MLP(input_dim=input_dims,
                output_dim=4,
                hidden_dim=1000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                val_freq=10,
                ncpu=ncpu)


    tb_logger = pl_loggers.TensorBoardLogger(f'act_{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_logs/')
    trainer   = pl.Trainer(gpus=[0], max_epochs=args.epoch, progress_bar_refresh_rate=20, logger=tb_logger)
    t         = time.time()
    trainer.fit(mlp, train_data_iter, valid_data_iter)
    print(time.time() - t, 's')

    print('Finish!')
