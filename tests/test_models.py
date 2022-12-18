from pathlib import Path

import pytest

from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.models.mlp import MLP

CHECKPOINT_ICLR_DIR = "checkpoints/iclr"


@pytest.fixture()
def iclr_checkpoint_files() -> list[Path]:
    return [
        find_best_model_ckpt(Path(CHECKPOINT_ICLR_DIR) / model)
        for model in "act rt1 rxn rt2".split()
    ]


@pytest.mark.skipif(
    not Path(CHECKPOINT_ICLR_DIR).exists(),  # assume if path exits, then all 4 files exist
    reason="ICLR checkpoints are not available",
)
def test_can_load_iclr_checkpoints(iclr_checkpoint_files):
    models = [load_mlp_from_ckpt(file) for file in iclr_checkpoint_files]
    assert len(models) == 4
    assert all(isinstance(model, MLP) for model in models)
