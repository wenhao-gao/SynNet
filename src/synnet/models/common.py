"""Common methods and params shared by all models.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml
from scipy import sparse

from synnet.encoding.distances import cosine_distance
from synnet.models.mlp import MLP
from synnet.MolEmbedder import MolEmbedder

logger = logging.getLogger(__file__)


def init_save_dir(path: str, suffix: str = "") -> Path:
    """Creates folder with timestamp: `$path/<timestamp>$suffix`."""
    from datetime import datetime

    now = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    save_dir = Path(path) / (now + suffix)

    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir


def load_config_file(file: str) -> dict[str, Union[str, int]]:
    """Load a `*.yaml`-config file."""
    file = Path(file)
    if not file.suffix == ".yaml":
        raise NotImplementedError(f"Can only read config from yaml file, not {file}.")
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def xy_to_dataloader(
    X_file: str,
    y_file: str,
    task: str,
    n: Union[int, float] = 1.0,
    **kwargs,
):
    """Loads featurized X,y `*.npz`-data into a `DataLoader`"""
    X = sparse.load_npz(X_file)
    y = sparse.load_npz(y_file)
    # Filer?
    if isinstance(n, int):
        n = min(n, X.shape[0])  # ensure n does not exceed size of dataset
        X = X[:n]
        y = y[:n]
    elif isinstance(n, float) and n < 1.0:
        xn = X.shape[0] * n
        yn = X.shape[0] * n
        X = X[:xn]
        y = y[:yn]
    else:
        pass  #
    X = np.atleast_2d(np.asarray(X.todense()))
    y = (
        np.atleast_2d(np.asarray(y.todense()))
        if task == "regression"
        else np.asarray(y.todense()).squeeze()
    )
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X),
        torch.Tensor(y),
    )
    logger.info(f"Loaded {X_file}, {X.shape=}")
    logger.info(f"Loaded {y_file}, {y.shape=}")
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _compute_class_weights_from_dataloader(dataloader, as_tensor: bool = False):
    from sklearn.utils.class_weight import compute_class_weight

    y: torch.Tensor = dataloader.dataset.tensors[-1]
    classes = y.unique().numpy()
    y = y.numpy()
    class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    if as_tensor:
        class_weight = torch.from_numpy(class_weight)
    return class_weight


def _fetch_molembedder(file: str):
    logger.info(f"Try to load precomputed MolEmbedder from {file}.")
    molembedder = MolEmbedder().load_precomputed(file).init_balltree(metric=cosine_distance)
    logger.info(f"Loaded MolEmbedder from {file}.")
    return molembedder


def load_mlp_from_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference."""
    try:
        model = MLP.load_from_checkpoint(ckpt_file)
    except TypeError:
        model = _load_mlp_from_iclr_ckpt(ckpt_file)
    return model.eval()


def find_best_model_ckpt(path: str) -> Union[Path, None]:
    """Find checkpoint with lowest val_loss.

    Poor man's regex:
    somepath/act/ckpts.epoch=70-val_loss=0.03.ckpt
                                         ^^^^--extract this as float
    """
    ckpts = Path(path).rglob("*.ckpt")
    best_model_ckpt = None
    lowest_loss = 10_000  # ~ math.inf
    for file in ckpts:
        stem = file.stem
        val_loss = float(stem.split("val_loss=")[-1])
        if val_loss < lowest_loss:
            best_model_ckpt = file
            lowest_loss = val_loss
    return best_model_ckpt


def _load_mlp_from_iclr_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference.
    Info: hparams were not saved, so we specify the ones needed for inference again."""
    model = Path(ckpt_file).parent.name  # assume "<dirs>/<model>/<file>.ckpt"
    kwargs = {
        "num_dropout_layers": 1,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "val_freq": 10,
    }
    if model == "act":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=3 * 4096,
            output_dim=4,
            hidden_dim=1000,
            num_layers=5,
            task="classification",
            dropout=0.5,
            loss="cross_entropy",
            valid_loss="accuracy",
            **kwargs,
        )
    elif model == "rt1":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=3 * 4096,
            output_dim=256,
            hidden_dim=1200,
            num_layers=5,
            task="regression",
            dropout=0.5,
            loss="mse",
            valid_loss="mse",  # Info: Used to be accuracy on kNN in embedding space, but that's very slow
            **kwargs,
        )
    elif model == "rxn":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=4 * 4096,
            output_dim=91,
            hidden_dim=3000,
            num_layers=5,
            task="classification",
            dropout=0.5,
            loss="mse",
            valid_loss="mse",  # Info: Used to be accuracy on kNN in embedding space, but that's very slow
            **kwargs,
        )
    elif model == "rt2":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=4 * 4096 + 91,
            output_dim=256,
            hidden_dim=3000,
            num_layers=5,
            task="regression",
            dropout=0.5,
            loss="mse",
            valid_loss="mse",  # Info: Used to be accuracy on kNN in embedding space, but that's very slow
            **kwargs,
        )

    else:
        raise ValueError
    return model.eval()


def asdict(obj) -> dict:
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("__")}


if __name__ == "__main__":
    pass
