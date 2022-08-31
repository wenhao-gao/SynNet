"""
Reactant2 network (for predicting 2nd reactant).
"""
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scipy import sparse

from syn_net.config import DATA_EMBEDDINGS_DIR, DATA_FEATURIZED_DIR
from syn_net.models.common import VALIDATION_OPTS, get_args, xy_to_dataloader
from syn_net.models.mlp import MLP, cosine_distance
from syn_net.MolEmbedder import MolEmbedder

logger = logging.getLogger(__name__)
MODEL_ID = Path(__file__).stem


def _fetch_molembedder():
    knn_embedding_id = validation_option[12:]
    file = Path(DATA_EMBEDDINGS_DIR) / f"enamine_us_emb_{knn_embedding_id}.npy"
    logger.info(f"Try to load precomputed MolEmbedder from {file}.")
    molembedder = MolEmbedder().load_precomputed(file).init_balltree(metric=cosine_distance)
    logger.info(f"Loaded MolEmbedder from {file}.")
    return molembedder


if __name__ == "__main__":

    args = get_args()

    validation_option = VALIDATION_OPTS[args.out_dim]

    # Get ID for the data to know what we're working with and find right files.
    id = (
        f"{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_{validation_option[12:]}/"
    )

    dataset = "train"
    train_dataloader = xy_to_dataloader(
        X_file=Path(DATA_FEATURIZED_DIR) / f"{id}/X_{MODEL_ID}_{dataset}.npz",
        y_file=Path(DATA_FEATURIZED_DIR) / f"{id}/y_{MODEL_ID}_{dataset}.npz",
        n=None if not args.debug else 1000,
        batch_size=args.batch_size,
        num_workers=args.ncpu,
        shuffle=True if dataset == "train" else False,
    )

    dataset = "valid"
    valid_dataloader = xy_to_dataloader(
        X_file=Path(DATA_FEATURIZED_DIR) / f"{id}/X_{MODEL_ID}_{dataset}.npz",
        y_file=Path(DATA_FEATURIZED_DIR) / f"{id}/y_{MODEL_ID}_{dataset}.npz",
        n=None if not args.debug else 1000,
        batch_size=args.batch_size,
        num_workers=args.ncpu,
        shuffle=True if dataset == "train" else False,
    )
    logger.info(f"Set up dataloaders.")

    # Fetch Molembedder and init BallTree
    molembedder = _fetch_molembedder()

    pl.seed_everything(0)
    INPUT_DIMS = {
        "fp": {
            "hb": int(4 * args.nbits + 91),
            "gin": int(4 * args.nbits + 4700),
        },
        "gin": {
            "hb": int(3 * args.nbits + args.out_dim + 91),
            "gin": int(3 * args.nbits + args.out_dim + 4700),
        },
    }  # somewhat constant...
    input_dims = INPUT_DIMS[args.featurize][args.rxn_template]

    mlp = MLP(
        input_dim=input_dims,
        output_dim=args.out_dim,
        hidden_dim=3000,
        num_layers=5,
        dropout=0.5,
        num_dropout_layers=1,
        task="regression",
        loss="mse",
        valid_loss=validation_option,
        optimizer="adam",
        learning_rate=1e-4,
        val_freq=10,
        molembedder=molembedder,
        ncpu=ncpu,
    )

    # Set up Trainer
    save_dir = Path(
        "results/logs/"
        + f"{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}"
        + f"/{MODEL_ID}"
    )
    save_dir.mkdir(exist_ok=True, parents=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_logger.log_dir,
        filename="ckpts.{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=10)

    max_epochs = args.epoch if not args.debug else 2
    # Create trainer
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=max_epochs,
        progress_bar_refresh_rate=int(len(train_data_iter) * 0.05),
        callbacks=[checkpoint_callback],
        logger=[tb_logger],
        fast_dev_run=True,
    )

    logger.info(f"Start training")
    trainer.fit(mlp, train_data_iter, valid_data_iter)
    logger.info(f"Training completed.")
