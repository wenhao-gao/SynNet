"""Reactant1 network (for predicting 1st reactant).
"""
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from synnet.models.common import get_args, load_config_file, xy_to_dataloader
from synnet.models.mlp import MLP

logger = logging.getLogger(__name__)
MODEL_ID = Path(__file__).stem

if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load config
    if args.config_file is None:
        setattr(args, "config_file", __file__.replace(".py", ".yaml"))
    config = load_config_file(args.config_file)
    logger.info(f"Config: {json.dumps(config,indent=2)}")

    pl.seed_everything(0)

    # Set up dataloaders
    train_dataloader = xy_to_dataloader(
        X_file=Path(config["data"]["Xtrain_file"]),
        y_file=Path(config["data"]["ytrain_file"]),
        batch_size=config["parameters"]["batch_size"],
        task=config["parameters"]["task"],
        num_workers=args.ncpu,
        n=None if not args.debug else 250,
        shuffle=True,
    )

    valid_dataloader = xy_to_dataloader(
        X_file=Path(config["data"]["Xvalid_file"]),
        y_file=Path(config["data"]["yvalid_file"]),
        batch_size=config["parameters"]["batch_size"],
        task=config["parameters"]["task"],
        num_workers=args.ncpu,
        n=None if not args.debug else 250,
        shuffle=False,
    )

    logger.info(f"Set up dataloaders.")

    # Fetch Molembedder and init BallTree
    molembedder = None  # _fetch_molembedder()

    mlp = MLP(**config["parameters"])

    # Set up Trainer
    result_dir = Path(config["result_dir"])
    result_dir.mkdir(exist_ok=True, parents=True)

    csv_logger = pl_loggers.CSVLogger(result_dir, name="")
    log_dir = csv_logger.log_dir
    logger.info(f"Log dir set to: {log_dir}")
    logger.info(f"Result dir dir set to: {result_dir}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=log_dir,
        filename="ckpts.{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=3)
    tqdm_callback = TQDMProgressBar(refresh_rate=max(1, int(len(train_dataloader) * 0.05)))

    wandb_logger = pl_loggers.WandbLogger(
        name=config["name"],
        project=config["project"],
        group=config["group"] + ("-debug" if args.debug else ""),
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config["parameters"]["max_epochs"] if not args.debug else 11,
        callbacks=[
            checkpoint_callback,
            tqdm_callback,
        ],
        logger=[csv_logger, wandb_logger],
        fast_dev_run=args.fast_dev_run,
    )

    logger.info(f"Start training")
    trainer.fit(mlp, train_dataloader, valid_dataloader)
    logger.info(f"Training completed.")
