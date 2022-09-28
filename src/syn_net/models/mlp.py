"""
Multi-layer perceptron (MLP) class.
"""
import logging
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from syn_net.MolEmbedder import MolEmbedder

logger = logging.getLogger(__name__)


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_dim=3072,
        output_dim=4,
        hidden_dim=1000,
        num_layers=5,
        dropout=0.5,
        num_dropout_layers=1,
        task="classification",
        loss="cross_entropy",
        valid_loss="accuracy",
        optimizer="adam",
        learning_rate=1e-4,
        val_freq=10,
        ncpu=16,
        molembedder: MolEmbedder = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="molembedder")
        self.loss = loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ncpu = ncpu
        self.val_freq = val_freq
        self.molembedder = molembedder

        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.ReLU())

        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.BatchNorm1d(hidden_dim))
            modules.append(nn.ReLU())
            if i > num_layers - 3 - num_dropout_layers:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_dim, output_dim))
        if task == "classification":
            modules.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        """Forward step for inference only."""
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        """The complete training loop."""
        x, y = batch
        y_hat = self.layers(x)
        if self.loss == "cross_entropy":
            loss = F.cross_entropy(y_hat, y.long())
        elif self.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.loss == "l1":
            loss = F.l1_loss(y_hat, y)
        elif self.loss == "huber":
            loss = F.huber_loss(y_hat, y)
        else:
            raise ValueError("Not specified loss function: % s" % self.loss)
        self.log(f"train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """The complete validation loop."""
        if self.trainer.current_epoch % self.val_freq == 0:
            x, y = batch
            y_hat = self.layers(x)
            if self.valid_loss == "cross_entropy":
                loss = F.cross_entropy(y_hat, y.long())
            elif self.valid_loss == "accuracy":
                y_hat = torch.argmax(y_hat, axis=1)
                accuracy = (y_hat == y).sum() / len(y)
                loss = 1 - accuracy
            elif self.valid_loss[:11] == "nn_accuracy":
                # NOTE: Very slow!
                # Performing the knn-search can easily take a couple of minutes,
                # even for small datasets.
                kdtree = self.molembedder.kdtree
                y = nn_search_list(y.detach().cpu().numpy(), None, kdtree)
                y_hat = nn_search_list(y_hat.detach().cpu().numpy(), None, kdtree)

                accuracy = (y_hat == y).sum() / len(y)
                loss = 1 - accuracy
            elif self.valid_loss == "mse":
                loss = F.mse_loss(y_hat, y)
            elif self.valid_loss == "l1":
                loss = F.l1_loss(y_hat, y)
            elif self.valid_loss == "huber":
                loss = F.huber_loss(y_hat, y)
            else:
                raise ValueError(
                    "Not specified validation loss function for '%s'" % self.valid_loss
                )
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            pass

    def configure_optimizers(self):
        """Define Optimerzers and LR schedulers."""
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_array(data_arrays, batch_size, is_train=True, ncpu=-1):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers=ncpu)


def cosine_distance(v1, v2, eps=1e-15):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2) + eps)


def nn_search_list(y, out_feat, kdtree):
    y = np.atleast_2d(y)  # (n_samples, n_features)
    ind = kdtree.query(y, k=1, return_distance=False)  # (n_samples, 1)
    return ind


if __name__ == "__main__":
    pass
