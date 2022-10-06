from syn_net.models.mlp import MLP


def load_mlp_from_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference."""
    model = MLP.load_from_checkpoint(ckpt_file)
    return model.eval()
