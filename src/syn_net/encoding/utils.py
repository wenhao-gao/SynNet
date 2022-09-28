import numpy as np


def one_hot_encoder(dim, space):
    """
    Create a one-hot encoded vector of length=`space`, with a non-zero element
    at the index given by `dim`.

    Args:
        dim (int): Non-zero bit in one-hot vector.
        space (int): Length of one-hot encoded vector.

    Returns:
        vec (np.ndarray): One-hot encoded vector.
    """
    vec = np.zeros((1, space))
    vec[0, dim] = 1
    return vec
