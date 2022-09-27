import logging
from pathlib import Path
from typing import Callable, Union

import numpy as np
from sklearn.neighbors import BallTree
from syn_net.config import MAX_PROCESSES
logger = logging.getLogger(__name__)


class MolEmbedder:
    def __init__(self, processes: int = MAX_PROCESSES) -> None:
        self.processes = processes
        self.func: Callable
        self.building_blocks: Union[list[str], np.ndarray]
        self.embeddings: np.ndarray
        self.kdtree: BallTree
        self.kdtree_metric: str

    def get_embeddings(self) -> np.ndarray:
        """Returns `self.embeddings` as 2d-array."""
        return np.atleast_2d(self.embeddings)

    def _compute_mp(self, data):
        from pathos import multiprocessing as mp

        with mp.Pool(processes=self.processes) as pool:
            embeddings = pool.map(self.func, data)
        return embeddings

    def compute_embeddings(self, func: Callable, building_blocks: list[str]):
        logger.info(f"Will compute embedding with {self.processes} processes.")
        logger.info(f"Embedding function: {func.__name__}")
        self.func = func
        if self.processes == 1:
            embeddings = list(map(self.func, building_blocks))
        else:
            embeddings = self._compute_mp(building_blocks)
        logger.info(f"Computed embeddings.")
        self.embeddings = embeddings
        return self

    def _save_npy(self, file: str):
        if self.embeddings is None:
            raise ValueError("Must have computed embeddings to save.")

        embeddings = np.asarray(self.embeddings)  # assume at least 2d
        np.save(file, embeddings)
        logger.info(f"Successfully saved data (shape={embeddings.shape}) to {file}.")
        return self

    def save_precomputed(self, file: str):
        """Saves pre-computed molecule embeddings to `*.npy`"""
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.suffixes == [".npy"]:
            self._save_npy(file)
        else:
            raise NotImplementedError(f"File have 'npy' extension, not {file.suffixes}")
        return self

    def _load_npy(self, file: Path):
        return np.load(file)

    def load_precomputed(self, file: str):
        """Loads a pre-computed molecule embeddings from `*.npy`"""
        file = Path(file)
        if file.suffixes == [".npy"]:
            self.embeddings = self._load_npy(file)
            self.kdtree = None
        else:
            raise NotImplementedError
        return self

    def init_balltree(self, metric: Union[Callable, str]):
        """Initializes a `BallTree`.

        Note:
            Can take a couple of minutes."""
        if self.embeddings is None:
            raise ValueError("Need emebddings to compute kdtree.")
        X = self.embeddings
        self.kdtree_metric = metric.__name__ if not isinstance(metric,str) else metric
        self.kdtree = BallTree(X, metric=metric)

        return self


