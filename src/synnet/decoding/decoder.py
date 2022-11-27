import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from synnet.data_generation.preprocessing import Reaction
from synnet.data_generation.syntrees import Encoder, MorganFingerprintEncoder, OneHotEncoder

logger = logging.getLogger(__name__)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import BallTree

from synnet.utils.data_utils import SyntheticTree, SyntheticTreeSet


class HelperDataloader:
    @classmethod
    def _fetch_data_chembl(cls, file: str) -> list[str]:
        df = pd.read_csv(file, sep="\t")
        smis_query = df["smiles"].to_list()
        return smis_query

    @classmethod
    def _fetch_data_from_file(cls, file: str) -> list[str]:
        with open(file, "rt") as f:
            smis_query = [line.strip() for line in f]
        return smis_query

    @classmethod
    def fetch_data(cls, file: str) -> list[str]:
        file = Path(file)
        if any(split in file.stem for split in ["train", "valid", "test"]):

            logger.info(f"Reading data from {file}")
            syntree_collection = SyntheticTreeSet().load(file)
            smiles = [syntree.root.smiles for syntree in syntree_collection]
        elif "chembl" in file.stem:
            smiles = cls._fetch_data_chembl(file)
        else:  # Hopefully got a filename instead
            smiles = cls._fetch_data_from_file(file)
        return smiles


class SynTreeDecoder:
    """Decoder for a molecular embedding."""

    def __init__(
        self,
        *,
        building_blocks: list[str],
        reaction_collection: list[Reaction],
        action_net: pl.LightningModule,
        reactant1_net: pl.LightningModule,
        rxn_net: pl.LightningModule,
        reactant2_net: pl.LightningModule,
        rxn_encoder: Encoder = OneHotEncoder(91),
        mol_encoder: Encoder = MorganFingerprintEncoder(2, 4096),
        building_blocks_embeddings: np.ndarray,  # TODO: can be fused with `balltree`
        balltree: BallTree,
        similarity_fct: Callable[[np.ndarray, List[str]], np.ndarray] = None,
    ) -> None:

        # Assets
        self.bblocks = building_blocks
        self.bblocks_emb = building_blocks_embeddings
        self.rxn_collection = reaction_collection
        self.num_reactions = len(self.rxn_collection)

        # Encoders
        self.rxn_encoder = rxn_encoder
        self.mol_encoder = mol_encoder

        # Networks
        self.nets: Dict[str, pl.LightningModule] = {
            k: v
            for k, v in zip(
                "act rt1 rxn rt2".split(), (action_net, reactant1_net, rxn_net, reactant2_net)
            )
        }

        # kNN search spaces
        self.balltree = balltree

        # Similarity fct
        self.similarity_fct = similarity_fct

        # Helper datastructures
        # A dict is used as lookup table for 2nd reactant during inference:
        self.bblocks_dict: Dict[str, int] = {block: i for i, block in enumerate(self.bblocks)}
        self.ACTIONS = {i: verb for i, verb in enumerate("add expand merge end".split())}

    def _get_syntree_state_embedding(self, state: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute state embedding for a state."""
        nbits: int = self.mol_encoder.nbits
        if len(state) == 0:
            z_mol_root1 = np.zeros(nbits)
            z_mol_root2 = np.zeros(nbits)
        elif len(state) == 1:  # Only one actively growing syntree
            z_mol_root1 = self.mol_encoder.encode(state[0])
            z_mol_root2 = np.zeros(nbits)
        elif len(state) == 2:  # Two syntrees
            z_mol_root1 = self.mol_encoder.encode(state[0])
            z_mol_root2 = self.mol_encoder.encode(state[1])
        else:
            raise ValueError(f"Unable to compute state embedding. Passed {state=}")

        return [z_mol_root1, z_mol_root2]

    def get_state_embedding(self, state: list[str], z_target: np.ndarray) -> np.ndarray:
        """Computes embeddings for all molecules in the input space.

        Embedding = [z_mol1, z_mol2, z_target]
        """
        z_mol_root1, z_mol_root2 = self._get_syntree_state_embedding(state)
        z_state = np.concatenate([z_mol_root1, z_mol_root2, z_target], axis=0)
        return z_state  # (d,)

    def _get_action_mask(self, syntree: SyntheticTree):
        """Get a mask of possible action for a SyntheticTree"""
        # Recall: (Add, Expand, Merge, and End)
        canAdd = False
        canMerge = False
        canExpand = False
        canEnd = False

        state = syntree.get_state()
        nTrees = len(state)
        if nTrees == 0:  # base case
            canAdd = True
        elif nTrees == 1 and (syntree.depth == self.max_depth - 1):
            # syntree is 1 update apart from its max depth, only allow to end it.
            canEnd = True
        elif nTrees == 1:
            canAdd = True
            canExpand = True
            canEnd = True
        elif nTrees == 2:
            canExpand = True  # TODO: do not expand when we're 2 steps away from max depth
            canMerge = any(self.get_reaction_mask(tuple(state)))
        else:
            raise ValueError

        return np.array((canAdd, canExpand, canMerge, canEnd), dtype=bool)

    def _find_valid_bimolecular_rxns(self, reactants: Tuple[str, str]) -> np.ndarray:
        reaction_mask: list[bool] = []
        for _rxn in self.rxn_collection.rxns:
            try:
                p = _rxn.run_reaction(reactants, allow_to_fail=False)
                is_valid_reaction = p is not None
            except Exception as e:
                # run_reactions() does some validity-checks and raises Exception
                is_valid_reaction = False
            reaction_mask += [is_valid_reaction]
        return np.asarray(reaction_mask)

    def _find_valid_unimolecular_rxns(self, reactant: str) -> np.ndarray:
        reaction_mask: list[bool] = []
        reaction_mask = [rxn.is_reactant(reactant) for rxn in self.rxn_collection.rxns]
        return np.asarray(reaction_mask)

    def get_reaction_mask(self, reactants: Union[str, Tuple[str, str]]) -> np.ndarray:
        if isinstance(reactants, str):
            return self._find_valid_unimolecular_rxns(reactants)
        else:
            return self._find_valid_bimolecular_rxns(reactants)

    def decode(
        self, z_target: np.ndarray, *, k_reactant1: int = 1, max_depth: int = 15, **kwargs
    ) -> dict[str, SyntheticTree]:
        if kwargs.get("debug"):
            logger.setLevel("DEBUG")
        eps = 1e-6  # small constant is added to probabilities so that masked values (=0.0) are unequal to probabilites (0+eps)
        self.max_depth = max_depth  # so we can access this param in methods
        act, rt1, rxn, rt2 = self.nets.values()

        syntree = SyntheticTree()
        mol_recent: Union[str, None] = None  # most-recent root mol
        i = 0
        while syntree.depth < self.max_depth:
            logger.debug(f"Iteration {i} | {syntree.depth=}")

            # Current state
            state = syntree.get_state()
            z_state = self.get_state_embedding(state, z_target)  # (3d,)
            z_state = torch.Tensor(z_state[None, :])  # (1,3d)

            # Prediction action
            p_action = act.forward(z_state)  # (1,4)
            p_action = p_action.detach().numpy() + eps
            action_mask = self._get_action_mask(syntree)
            action_id = np.argmax(p_action * action_mask)
            logger.debug(f" Action: {self.ACTIONS[action_id]}. ({p_action.round(2)=})")

            if self.ACTIONS[action_id] == "end":
                break
            elif self.ACTIONS[action_id] == "add":
                # Start a new sub-syntree.
                z_reactant1 = rt1.forward(z_state)  # TODO: z=z' as mol embedding dim is differnt
                z_reactant1 = z_reactant1.detach().numpy()  # (1,d')

                # Select building block via kNN search
                k = k_reactant1 if i == 0 else 1
                logger.debug(f"  k-NN search for 1st reactant with k={k}.")
                idxs = self.balltree.query(z_reactant1, k=k, return_distance=False)
                # idxs.shape = (1,k)
                idx = idxs[0][k - 1]
                reactant_1: str = self.bblocks[idx]
                logger.debug(f"  Selected 1st reactant ({idx=}): `{reactant_1}`")
            elif self.ACTIONS[action_id] == "expand":
                # We already have a 1st reactant.
                reactant_1: str = mol_recent  # aka root mol (=product) from last iteration
            elif self.ACTIONS[action_id] == "merge":
                # We already have 1st + 2nd reactant
                # TODO: If we merge two trees, we have to determine a reaction.
                #       This means we have to encode the "1st reactant"
                #       -> Investitage if this is
                #           - the `mol_recent`
                #           - the root mol of the "other" syntree
                #       Note: It does not matter for the reaction, but it does
                #             matter for the reaction prediction.
                reactant_1 = mol_recent
                pass

            # Predict reaction
            z_reactant_1 = self.mol_encoder.encode(reactant_1)  # (d,)
            z_reactant_1 = z_reactant_1[None, :]  # (1,d)

            x = np.concatenate((z_state, z_reactant_1), axis=1)
            x = torch.Tensor(x)
            p_rxn = rxn.forward(x)
            p_rxn = p_rxn.detach().numpy() + eps
            logger.debug(
                "  Top 5 reactions: "
                + ", ".join(
                    [
                        f"{__idx:>2d} (p={p_rxn[0][__idx]:.2f})"
                        for __idx in np.argsort(p_rxn)[0, -5:][::-1]
                    ]
                )
            )

            # Reaction mask
            if self.ACTIONS[action_id] == "merge":
                reactant_2 = (set(state) - set([reactant_1])).pop()
                # TODO: fix these shenanigans and determine reliable which is the 2nd reactant
                #       by "knowing" the order of the state
                reaction_mask = self.get_reaction_mask(
                    (reactant_1, reactant_2)
                )  # if merge, only allow bi-mol rxn
            else:  # add or expand (both start from 1 reactant only)
                reaction_mask = self.get_reaction_mask(reactant_1)
            logger.debug(f"  Reaction mask with n choices: {reaction_mask.sum()}")

            # Check: Are any reactions possible? If not, break. # TODO: Cleanup
            if not any(reaction_mask):
                # Is it possible to gracefully end this tree?
                # If there is only a sinlge tree, mark it as "ended"
                if len(state) == 1:
                    action_id = 3
                logger.debug(
                    f"Terminated decoding as no reaction is possible, manually enforced {action_id=} "
                )
                break

            # Select reaction template
            rxn_id = np.argmax(p_rxn * reaction_mask)
            reaction: Reaction = self.rxn_collection.rxns[rxn_id]  # TODO: fix why we need type hint
            logger.debug(
                f"  Selected {'bi' if reaction.num_reactant==2 else 'uni'} reaction {rxn_id=}"
            )

            # We have three options:
            #  1. "merge" -> need to sample 2nd reactant
            #  2. "expand" or "expand" -> only sample 2nd reactant if reaction is bimol
            if self.ACTIONS[action_id] == "merge":
                reactant_2 = syntree.get_state()[1]  # "old" root mol, i.e. other in state
            elif self.ACTIONS[action_id] in ["add", "expand"]:
                if reaction.num_reactant == 2:
                    # Sample 2nd reactant
                    z_rxn = self.rxn_encoder.encode(rxn_id)
                    x = np.concatenate((z_state, z_reactant_1, z_rxn), axis=1)
                    x = torch.Tensor(x)  # (1,16475)

                    z_reactant2 = rt2.forward(x)
                    z_reactant2 = z_reactant2.detach().numpy()

                    # Select building block via kNN search
                    # does reactant 1 match position 0 or 1 in the template?
                    if reaction.is_reactant_first(reactant_1):
                        # TODO: can match pos 1 AND 2 at teh same time
                        available_reactants_2 = reaction.available_reactants[1]
                    else:
                        available_reactants_2 = reaction.available_reactants[0]

                    # Get smiles -> index -> embedding
                    _idx = [self.bblocks_dict[_smiles] for _smiles in available_reactants_2]
                    _emb = self.bblocks_emb[
                        _idx
                    ]  # TODO: Check if ordering is correct -> Seems legit
                    logger.debug(f"  Subspace of available 2nd reactants: {len(_idx)} ")
                    _dists = cosine_distances(_emb, z_reactant2)
                    idx = np.argmin(_dists)  # 1.5-5x faster WORTH ITTTT 🥳🪅

                    # _balltree = BallTree(_emb, metric=cosine_distance)
                    # k = 1
                    # logger.debug(f"  k-NN search for 2nd reactant with k={k}.")
                    # idxs = _balltree.query(z_reactant2, k=k, return_distance=False)
                    # # idxs.shape = (1,k)
                    # idx = idxs[0][k - 1]
                    reactant_2: str = available_reactants_2[idx]
                    logger.debug(f"  Selected 2nd reactant ({idx=}): `{reactant_2}`")
                else:  # this is a unimolecular reaction
                    reactant_2 = None

            # Run reaction
            product: str = reaction.run_reaction((reactant_1, reactant_2), allow_to_fail=False)
            logger.debug(f"  Ran reaction {reactant_1} + {reactant_2} -> {product}")

            # Validate outcome of reaction
            if product is None:
                error_msg = (
                    f"rdkit.RunReactants() produced invalid product. "
                    + f"Reaction ID: {rxn_id}, {syntree.depth=} "
                    + f"Reaction: `{reactant_1} + {reactant_2} -> {product}`"
                )
                # logger.error("  " + error_msg)
                syntree._log = error_msg
                # Is it possible to gracefully end this tree?
                # If there is only a sinlge tree, mark it as "ended"
                if len(state) == 1:
                    action_id = 3
                logger.debug(
                    f"Terminated decoding as no reaction is possible, manually enforced {action_id=} "
                )
                break

            # Update
            logger.debug("  Updating SynTree.")
            syntree.update(int(action_id), int(rxn_id), reactant_1, reactant_2, product)
            mol_recent = product
            i += 1

        # End of generation. Validate outcome
        if self.ACTIONS[action_id] == "end":
            syntree.update(int(action_id), None, None, None, None)

        # Compute similarity to target
        if syntree.is_valid and self.similarity_fct is not None:
            similarities = self.compute_similarity_to_target(
                similarity_fct=self.similarity_fct,
                z_target=z_target,
                syntree=syntree,
            )
        else:
            similarities = None

        return {"syntree": syntree, "max_similarity": np.max(similarities)}

    def compute_similarity_to_target(
        self,
        *,
        similarity_fct: Callable[[np.ndarray, List[str]], np.ndarray],
        z_target: np.ndarray,
        syntree: SyntheticTree,
    ) -> np.ndarray:  # TODO: move to its own class?
        """Computes the similarity to a `z_target` for all nodes, as
        we can in theory truncate the tree to our liking.
        """
        return np.array(similarity_fct(z_target, [smi for smi in syntree.nodes_as_smiles]))


class SynTreeDecoderGreedy:
    def __init__(self, decoder: SynTreeDecoder) -> None:
        self.decoder = decoder  # composition over inheritance

    def decode(
        self,
        z_target: np.ndarray,
        *,
        attempts: int = 3,
        objective: Optional[str] = "best",  # "best", "best+shortest"
        **decode_kwargs: dict,
    ) -> SyntheticTree:
        """Decode `z_target` at most `attempts`-times and return the most-similar one."""

        best_similarity = -np.inf
        best_syntree = SyntheticTree()
        for i in range(attempts):
            logger.debug(f"Greedy search attempt: {i} (k_reactant1={i+1})")

            res = self.decoder.decode(z_target, k_reactant1=i + 1, **decode_kwargs)

            syntree = res.get("syntree")
            max_similarity = res.get("max_similarity")

            #  ↓ for legacy decoder, which could return None
            if syntree is None or not syntree.is_valid:
                continue
            # Sanity check:
            if not max_similarity:
                raise ValueError("Did you specify a `similarity_fct` for the decoder?")

            # Do we have a new best candidate?
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_syntree = syntree
            logger.debug(f"  Max similarity: {max_similarity:.3f} (best: {best_similarity:.3f})")
            if objective == "best" and best_similarity == 1.0:
                logger.debug(
                    f"Decoded syntree has similarity 1.0 and {objective=}; abort greedy search."
                )
                break

        # Return best results
        return {"syntree": best_syntree, "max_similarity": best_similarity, "attempts": i + 1}
