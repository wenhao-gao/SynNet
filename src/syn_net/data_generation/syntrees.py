"""syntrees
"""
import logging
from typing import Tuple, Union

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from syn_net.config import MAX_PROCESSES

logger = logging.getLogger(__name__)

from syn_net.utils.data_utils import Reaction, SyntheticTree


class NoReactantAvailableError(Exception):
    """No second reactant available for the bimolecular reaction."""

    def __init__(self, message):
        super().__init__(message)


class NoReactionAvailableError(Exception):
    """Reactant does not match any reaction template."""

    def __init__(self, message):
        super().__init__(message)


class NoBiReactionAvailableError(Exception):
    """Reactants do not match any reaction template."""

    def __init__(self, message):
        super().__init__(message)


class NoReactionPossibleError(Exception):
    """`rdkit` can not yield a valid reaction product."""

    def __init__(self, message):
        super().__init__(message)

class MaxDepthError(Exception):
    """Synthetic Tree has exceeded its maximum depth."""

    def __init__(self, message):
        super().__init__(message)


class SynTreeGenerator:

    building_blocks: list[str]
    rxn_templates: list[Reaction]
    rxns: list[Reaction]
    IDX_RXNS: np.ndarray  # (nReactions,)
    ACTIONS: dict[int, str] = {i: action for i, action in enumerate("add expand merge end".split())}
    verbose: bool

    def __init__(
        self,
        *,
        building_blocks: list[str],
        rxn_templates: list[str],
        rng=np.random.default_rng(),
        processes: int = MAX_PROCESSES,
        verbose: bool = False,
    ) -> None:
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates
        self.rxns = [Reaction(template=tmplt) for tmplt in rxn_templates]
        self.rng = rng
        self.IDX_RXNS = np.arange(len(self.rxns))
        self.processes = processes
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)

        # Time intensive tasks
        self._init_rxns_with_reactants()

    def __match_mp(self):
        # TODO: refactor / merge with `BuildingBlockFilter`
        # TODO: Rename `ReactionSet` -> `ReactionCollection` (same for `SyntheticTreeSet`)
        #       `Reaction` as "datacls", `*Collection` as cls that encompasses operations on "data"?
        #       Third class simpyl for file I/O or include somewhere?
        from functools import partial

        from pathos import multiprocessing as mp

        def __match(bblocks: list[str], _rxn: Reaction):
            return _rxn.set_available_reactants(bblocks)

        func = partial(__match, self.building_blocks)
        with mp.Pool(processes=self.processes) as pool:
            rxns = pool.map(func, self.rxns)

        self.rxns = rxns
        return self

    def _init_rxns_with_reactants(self):
        """Initializes a `Reaction` with a list of possible reactants.

        Info: This can take a while for lots of possible reactants."""
        self.rxns = tqdm(self.rxns) if self.verbose else self.rxns
        if self.processes == 1:
            self.rxns = [rxn.set_available_reactants(self.building_blocks) for rxn in self.rxns]
        else:
            self.__match_mp()

        self.rxns_initialised = True
        return self

    def _sample_molecule(self) -> str:
        """Sample a molecule."""
        idx = self.rng.choice(len(self.building_blocks))
        smiles = self.building_blocks[idx]
        logger.debug(f"    Sampled molecule: {smiles}")
        return smiles

    def _base_case(self) -> str:
        return self._sample_molecule()

    def _find_rxn_candidates(self, smiles: str, raise_exc: bool = True) -> list[bool]:
        """Tests which reactions have `mol` as reactant."""
        mol = Chem.MolFromSmiles(smiles)
        rxn_mask = [rxn.is_reactant(mol) for rxn in self.rxns]
        if raise_exc and not any(rxn_mask):  # Do not raise exc when checking if two mols can react
            raise NoReactionAvailableError(f"No reaction available for: {smiles}.")
        return rxn_mask

    def _sample_rxn(self, mask: np.ndarray = None) -> Tuple[Reaction, int]:
        """Sample a reaction by index."""
        if mask is None:
            irxn_mask = self.IDX_RXNS  # All reactions are possible
        else:
            mask = np.asarray(mask)
            irxn_mask = self.IDX_RXNS[mask]
        idx = self.rng.choice(irxn_mask)
        logger.debug(
            f"Sampled reaction with index: {idx} (nreactants: {self.rxns[idx].num_reactant})"
        )
        return self.rxns[idx], idx

    def _expand(self, reactant_1: str) -> Tuple[str, str, str, np.int64]:
        """Expand a sub-tree from one molecule.
        This can result in uni- or bimolecular reaction."""

        # Identify applicable reactions
        rxn_mask = self._find_rxn_candidates(reactant_1)

        # Sample reaction (by index)
        rxn, idx_rxn = self._sample_rxn(mask=rxn_mask)

        # Sample 2nd reactant
        if rxn.num_reactant == 1:
            reactant_2 = None
        else:
            # Sample a molecule from the available reactants of this reaction
            # That is, for a reaction A + B -> C,
            #   - determine if we have "A" or "B"
            #   - then sample "B" (or "A")
            idx = 1 if rxn.is_reactant_first(reactant_1) else 0
            available_reactants = rxn.available_reactants[idx]
            nPossibleReactants = len(available_reactants)
            if nPossibleReactants == 0:
                raise NoReactantAvailableError(
                    f"Unable to find reactant {idx+1} for bimolecular reaction (ID: {idx_rxn}) and reactant {reactant_1}."
                )
                # TODO: 2 bi-molecular rxn templates have no matching bblock
            # TODO: use numpy array to avoid type conversion or stick to sampling idx?
            idx = self.rng.choice(nPossibleReactants)
            reactant_2 = available_reactants[idx]

        # Run reaction
        reactants = (reactant_1, reactant_2)
        product = rxn.run_reaction(reactants)
        return *reactants, product, idx_rxn

    def _get_action_mask(self, syntree: SyntheticTree):
        """Get a mask of possible action for a SyntheticTree"""
        # Recall: (Add, Expand, Merge, and End)
        canAdd = False
        canMerge = False
        canExpand = False
        canEnd = False

        state = syntree.get_state()
        nTrees = len(state)
        if nTrees == 0:
            canAdd = True
        elif nTrees == 1:
            canAdd = True
            canExpand = True
            canEnd = True # TODO: When syntree has reached max depth, only allow to end it.
        elif nTrees == 2:
            canExpand = True
            canMerge = any(self._get_rxn_mask(tuple(state)))
        else:
            raise ValueError

        return np.array((canAdd, canExpand, canMerge, canEnd), dtype=bool)

    def _get_rxn_mask(self, reactants: tuple[str, str]) -> list[bool]:
        """Get a mask of possible reactions for the two reactants."""
        masks = [self._find_rxn_candidates(r, raise_exc=False) for r in reactants]
        # TODO: We do not check if the two reactants are 1st and 2nd reactants in a given reaction.
        #       It is possible that both are only applicable as 1st reactant,
        #       and then the reaction is not possible, although the mask returns true.
        #       Alternative: Run the reaction and check if the product is valid.
        mask = [rxn1 and rxn2 for rxn1, rxn2 in zip(*masks)]
        if not any(mask):
            raise NoBiReactionAvailableError(f"No reaction available for {reactants}.")
        return mask

    def generate(self, max_depth: int = 15, retries: int = 3):
        """Generate a syntree by random sampling."""

        # Init
        logger.debug(f"Starting synthetic tree generation with {max_depth=} ")
        syntree = SyntheticTree()
        recent_mol = self._sample_molecule()  # root of the current tree

        for i in range(max_depth):
            logger.debug(f"Iteration {i}")

            # State of syntree
            state = syntree.get_state()

            # Sample action
            p_action = self.rng.random((1, 4))  # (1,4)
            action_mask = self._get_action_mask(syntree)  # (1,4)
            act = np.argmax(p_action * action_mask)  # (1,)
            action = self.ACTIONS[act]
            logger.debug(f"  Sampled action: {action}")

            if action == "end":
                r1, r2, p, idx_rxn = None, None, None, -1
            elif action == "expand":
                for j in range(retries):
                    logger.debug(f"    Try {j}")
                    r1, r2, p, idx_rxn = self._expand(recent_mol)
                    if p is not None:
                        break
                if p is None:
                    # TODO: move to rxn.run_reaction?
                    raise NoReactionPossibleError(
                        f"Reaction (ID: {idx_rxn}) not possible with: {r1} + {r2}."
                    )
            elif action == "add":
                mol = self._sample_molecule()
                r1, r2, p, idx_rxn = self._expand(mol)
                # Expand this subtree: reactant, reaction, reactant2
            elif action == "merge":
                # merge two subtrees: sample reaction, run it.

                # Identify suitable rxn
                r1, r2 = syntree.get_state()
                rxn_mask = self._get_rxn_mask(tuple((r1, r2)))
                # Sample reaction
                rxn, idx_rxn = self._sample_rxn(mask=rxn_mask)
                # Run reaction
                p = rxn.run_reaction((r1, r2))
                if p is None:
                    # TODO: move to rxn.run_reaction?
                    raise NoReactionPossibleError(
                        f"Reaction (ID: {idx_rxn}) not possible with: {r1} + {r2}."
                    )
            else:
                raise ValueError(f"Invalid action {action}")

            # Prepare next iteration
            logger.debug(f"    Ran reaction {r1} + {r2} -> {p}")

            recent_mol = p

            # Update tree
            syntree.update(act, rxn_id=int(idx_rxn), mol1=r1, mol2=r2, mol_product=p)
            logger.debug(f"SynTree updated.")
            if action == "end":
                break

        if i==max_depth-1 and not action == "end":
            raise MaxDepthError("Maximum depth {max_depth} exceeded.")
        logger.debug(f"🙌 SynTree completed.")
        return syntree


def wraps_syntreegenerator_generate(
    stgen: SynTreeGenerator,
) -> Tuple[Union[SyntheticTree, None], Union[Exception, None]]:
    """Wrapper for `SynTreeGenerator().generate` that catches all Exceptions."""
    try:
        st = stgen.generate()
    except NoReactantAvailableError as e:
        logger.error(e)
        return None, e
    except NoReactionAvailableError as e:
        logger.error(e)
        return None, e
    except NoBiReactionAvailableError as e:
        logger.error(e)
        return None, e
    except NoReactionPossibleError as e:
        logger.error(e)
        return None, e
    except TypeError as e:
        # When converting an invalid molecule from SMILES to rdkit Molecule.
        # This happens if the reaction template/rdkit produces an invalid product.
        logger.error(e)
        return None, e
    except Exception as e:
        logger.error(e, exc_info=e, stack_info=False)
        return None, e
    else:
        return st, None


def load_syntreegenerator(file: str) -> SynTreeGenerator:
    import pickle

    with open(file, "rb") as f:
        syntreegenerator = pickle.load(f)
    return syntreegenerator


def save_syntreegenerator(syntreegenerator: SynTreeGenerator, file: str) -> None:
    import pickle

    with open(file, "wb") as f:
        pickle.dump(syntreegenerator, f)
