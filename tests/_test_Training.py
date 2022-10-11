"""
Unit tests for model training.
"""
from pathlib import Path
import unittest
import shutil
from multiprocessing import cpu_count
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from scipy import sparse
import torch

from synnet.models.mlp import MLP, load_array
from synnet.MolEmbedder import MolEmbedder


TEST_DIR = Path(__file__).parent

REACTION_TEMPLATES_FILE =  f"{TEST_DIR}/assets/rxn_set_hb_test.txt"

def _fetch_molembedder():
    file = "tests/data/building_blocks_emb.npy"
    molembedder = MolEmbedder().load_precomputed(file).init_balltree(metric="euclidean")
    return molembedder

class TestReactionTemplateFile(unittest.TestCase):

    def test_number_of_reaction_templates(self):
        """ Count number of lines in file, i.e. the number of reaction templates."""
        with open(REACTION_TEMPLATES_FILE,"r") as f:
            nReactionTemplates = sum(1 for _ in f)
        self.assertEqual(nReactionTemplates,3)


class TestTraining(unittest.TestCase):
    """
    Tests for model training: (1) action network, (2) reactant 1 network, (3)
    reaction network, (4) reactant 2 network.
    """

    def setUp(self) -> None:
        import warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings("ignore", ".*GPU available but not used.*")

    def test_action_network(self):
        """
        Tests the Action Network.
        """
        embedding = "fp"
        radius = 2
        nbits = 4096
        batch_size = 10
        epochs = 2
        ncpu = min(2,cpu_count())
        validation_option = "accuracy"
        ref_dir = f"{TEST_DIR}/data/ref/"

        X = sparse.load_npz(ref_dir + "X_act_train.npz")
        assert X.shape==(4,3*nbits) # (4,12288)
        y = sparse.load_npz(ref_dir + "y_act_train.npz")
        assert y.shape==(4,1) # (4,1)
        X = torch.Tensor(X.A)
        y = torch.LongTensor(
            y.A.reshape(
                -1,
            )
        )
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(
            input_dim=int(3 * nbits),
            output_dim=4,
            hidden_dim=100,
            num_layers=3,
            dropout=0.5,
            num_dropout_layers=1,
            task="classification",
            loss="cross_entropy",
            valid_loss=validation_option,
            optimizer="adam",
            learning_rate=1e-4,
            val_freq=10,
            ncpu=ncpu,
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            f"act_{embedding}_{radius}_{nbits}_logs/"
        )
        trainer = pl.Trainer(
            max_epochs=epochs, logger=tb_logger, weights_summary=None,
        )
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 1.2967982292175293

        shutil.rmtree(f"act_{embedding}_{radius}_{nbits}_logs/")
        self.assertAlmostEqual(train_loss, train_loss_ref)

    def test_reactant1_network(self):
        """
        Tests the Reactant 1 Network.
        """
        embedding = "fp"
        radius = 2
        nbits = 4096
        out_dim = 300  # Note: out_dim 300 = gin embedding
        batch_size = 10
        epochs = 2
        ncpu = min(2,cpu_count())
        validation_option = "nn_accuracy_gin_unittest"
        ref_dir = f"{TEST_DIR}/data/ref/"

        # load the reaction data
        X = sparse.load_npz(ref_dir + "X_rt1_train.npz")
        assert X.shape==(2,3*nbits) # (4,12288)
        X = torch.Tensor(X.A)
        y = sparse.load_npz(ref_dir + "y_rt1_train.npz")
        assert y.shape==(2,300) # (2,300)
        y = torch.Tensor(y.A)
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(
            input_dim=int(3 * nbits),
            output_dim=out_dim,
            hidden_dim=100,
            num_layers=3,
            dropout=0.5,
            num_dropout_layers=1,
            task="regression",
            loss="mse",
            valid_loss=validation_option,
            optimizer="adam",
            learning_rate=1e-4,
            val_freq=10,
            molembedder=_fetch_molembedder(),
            ncpu=ncpu,
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            f"rt1_{embedding}_{radius}_{nbits}_logs/"
        )
        trainer = pl.Trainer(
            max_epochs=epochs, logger=tb_logger, weights_summary=None,
        )
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 0.33368119597435

        shutil.rmtree(f"rt1_{embedding}_{radius}_{nbits}_logs/")
        self.assertAlmostEqual(train_loss, train_loss_ref)

    def test_reaction_network(self):
        """
        Tests the Reaction Network.
        """
        embedding = "fp"
        radius = 2
        nbits = 4096
        batch_size = 10
        epochs = 2
        ncpu = min(2,cpu_count())
        n_templates = 3  # num templates in `REACTION_TEMPLATES_FILE`
        validation_option = "accuracy"
        ref_dir = f"{TEST_DIR}/data/ref/"

        X = sparse.load_npz(ref_dir + "X_rxn_train.npz")
        assert X.shape==(2,4*nbits) # (2,16384)
        y = sparse.load_npz(ref_dir + "y_rxn_train.npz")
        assert y.shape==(2, 1) # (2, 1)
        X = torch.Tensor(X.A)
        y = torch.LongTensor(
            y.A.reshape(
                -1,
            )
        )
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(
            input_dim=int(4 * nbits),
            output_dim=n_templates,
            hidden_dim=100,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task="classification",
            loss="cross_entropy",
            valid_loss=validation_option,
            optimizer="adam",
            learning_rate=1e-4,
            val_freq=10,
            ncpu=ncpu,
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            f"rxn_{embedding}_{radius}_{nbits}_logs/"
        )
        trainer = pl.Trainer(
            max_epochs=epochs, logger=tb_logger, weights_summary=None,
        )
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 1.1214743852615356

        shutil.rmtree(f"rxn_{embedding}_{radius}_{nbits}_logs/")
        self.assertAlmostEqual(train_loss, train_loss_ref,places=-6)

    def test_reactant2_network(self):
        """
        Tests the Reactant 2 Network.
        """
        embedding = "fp"
        radius = 2
        nbits = 4096
        out_dim = 300  # Note: out_dim 300 = gin embedding
        batch_size = 10
        epochs = 2
        ncpu = min(2,cpu_count())
        n_templates = 3  # num templates in 'data/rxn_set_hb_test.txt'
        validation_option = "nn_accuracy_gin_unittest"
        ref_dir = f"{TEST_DIR}/data/ref/"

        X = sparse.load_npz(ref_dir + "X_rt2_train.npz")
        assert X.shape==(2,4*nbits+n_templates) # (2,16387)
        y = sparse.load_npz(ref_dir + "y_rt2_train.npz")
        assert y.shape==(2,300) # (2,300)
        X = torch.Tensor(X.A)
        y = torch.Tensor(y.A)
        train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

        # use the train data for validation too (just for the unit tests)
        valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

        pl.seed_everything(0)
        mlp = MLP(
            input_dim=int(4 * nbits + n_templates),
            output_dim=out_dim,
            hidden_dim=100,
            num_layers=3,
            dropout=0.5,
            num_dropout_layers=1,
            task="regression",
            loss="mse",
            valid_loss=validation_option,
            optimizer="adam",
            learning_rate=1e-4,
            val_freq=10,
            molembedder=_fetch_molembedder(),
            ncpu=ncpu,
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            f"rt2_{embedding}_{radius}_{nbits}_logs/"
        )
        trainer = pl.Trainer(
            max_epochs=epochs, logger=tb_logger, weights_summary=None,
        )
        trainer.fit(mlp, train_data_iter, valid_data_iter)

        train_loss = float(trainer.callback_metrics["train_loss"])
        train_loss_ref = 0.3026905953884125

        shutil.rmtree(f"rt2_{embedding}_{radius}_{nbits}_logs/")
        self.assertAlmostEqual(train_loss, train_loss_ref)
