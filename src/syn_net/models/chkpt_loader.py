from typing import Tuple
from syn_net.models.mlp import MLP
import pytorch_lightning as pl
from typing import List

def load_modules_from_checkpoint(
    path_to_act: str,
    path_to_rt1: str,
    path_to_rxn: str,
    path_to_rt2: str,
    featurize: str,
    rxn_template: str,
    out_dim: int,
    nbits: int,
    ncpu: int,
) -> List[pl.LightningModule]:

    if rxn_template == "unittest":

        act_net = MLP.load_from_checkpoint(
            path_to_act,
            input_dim=int(3 * nbits),
            output_dim=4,
            hidden_dim=100,
            num_layers=3,
            dropout=0.5,
            num_dropout_layers=1,
            task="classification",
            loss="cross_entropy",
            valid_loss="accuracy",
            optimizer="adam",
            learning_rate=1e-4,
            ncpu=ncpu,
        )

        rt1_net = MLP.load_from_checkpoint(
            path_to_rt1,
            input_dim=int(3 * nbits),
            output_dim=out_dim,
            hidden_dim=100,
            num_layers=3,
            dropout=0.5,
            num_dropout_layers=1,
            task="regression",
            loss="mse",
            valid_loss="mse",
            optimizer="adam",
            learning_rate=1e-4,
            ncpu=ncpu,
        )

        rxn_net = MLP.load_from_checkpoint(
            path_to_rxn,
            input_dim=int(4 * nbits),
            output_dim=3,
            hidden_dim=100,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task="classification",
            loss="cross_entropy",
            valid_loss="accuracy",
            optimizer="adam",
            learning_rate=1e-4,
            ncpu=ncpu,
        )

        rt2_net = MLP.load_from_checkpoint(
            path_to_rt2,
            input_dim=int(4 * nbits + 3),
            output_dim=out_dim,
            hidden_dim=100,
            num_layers=3,
            dropout=0.5,
            num_dropout_layers=1,
            task="regression",
            loss="mse",
            valid_loss="mse",
            optimizer="adam",
            learning_rate=1e-4,
            ncpu=ncpu,
        )
    elif featurize == "fp":

        act_net = MLP.load_from_checkpoint(
            path_to_act,
            input_dim=int(3 * nbits),
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
            ncpu=ncpu,
        )

        rt1_net = MLP.load_from_checkpoint(
            path_to_rt1,
            input_dim=int(3 * nbits),
            output_dim=int(out_dim),
            hidden_dim=1200,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task="regression",
            loss="mse",
            valid_loss="mse",
            optimizer="adam",
            learning_rate=1e-4,
            ncpu=ncpu,
        )

        if rxn_template == "hb":

            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(4 * nbits),
                output_dim=91,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="classification",
                loss="cross_entropy",
                valid_loss="accuracy",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(4 * nbits + 91),
                output_dim=int(out_dim),
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="regression",
                loss="mse",
                valid_loss="mse",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

        elif rxn_template == "pis":

            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(4 * nbits),
                output_dim=4700,
                hidden_dim=4500,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="classification",
                loss="cross_entropy",
                valid_loss="accuracy",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(4 * nbits + 4700),
                output_dim=out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="regression",
                loss="mse",
                valid_loss="mse",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

    elif featurize == "gin":

        act_net = MLP.load_from_checkpoint(
            path_to_act,
            input_dim=int(2 * nbits + out_dim),
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
            ncpu=ncpu,
        )

        rt1_net = MLP.load_from_checkpoint(
            path_to_rt1,
            input_dim=int(2 * nbits + out_dim),
            output_dim=out_dim,
            hidden_dim=1200,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task="regression",
            loss="mse",
            valid_loss="mse",
            optimizer="adam",
            learning_rate=1e-4,
            ncpu=ncpu,
        )

        if rxn_template == "hb":

            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(3 * nbits + out_dim),
                output_dim=91,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="classification",
                loss="cross_entropy",
                valid_loss="accuracy",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(3 * nbits + out_dim + 91),
                output_dim=out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="regression",
                loss="mse",
                valid_loss="mse",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

        elif rxn_template == "pis":

            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(3 * nbits + out_dim),
                output_dim=4700,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="classification",
                loss="cross_entropy",
                valid_loss="accuracy",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(3 * nbits + out_dim + 4700),
                output_dim=out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task="regression",
                loss="mse",
                valid_loss="mse",
                optimizer="adam",
                learning_rate=1e-4,
                ncpu=ncpu,
            )

    act_net.eval()
    rt1_net.eval()
    rxn_net.eval()
    rt2_net.eval()

    return act_net, rt1_net, rxn_net, rt2_net
