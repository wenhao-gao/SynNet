"""syntrees
"""
from typing import Tuple
from tqdm import tqdm
import numpy as np
from rdkit import Chem

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from syn_net.utils.data_utils import Reaction, SyntheticTree

class NoReactantAvailableError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

class NoReactionAvailableError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

class NoReactionPossible(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class SynTreeGenerator:

    building_blocks: list[str]
    rxn_templates: list[Reaction]
    rxns: dict[int, Reaction]
    IDX_RXNS: list
    ACTIONS: dict[int, str] = {i: action for i, action in enumerate("add expand merge end".split())}
    verbose: bool

    def __init__(
        self,
        *,
        building_blocks: list[str],
        rxn_templates: list[str],
        rng=np.random.default_rng(seed=42),
        verbose:bool = False,
    ) -> None:
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates
        self.rxns = [Reaction(template=tmplt) for tmplt in rxn_templates]
        self.rng = rng
        self.IDX_RXNS = np.arange(len(self.rxns))
        self.processes = 32
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

    def _find_rxn_candidates(self, smiles: str):
        """Find a reaction with `mol` as reactant."""
        mol = Chem.MolFromSmiles(smiles)
        rxn_mask = [rxn.is_reactant(mol) for rxn in self.rxns]
        if not any(rxn_mask):
            raise NoReactionAvailableError(f"No reaction available for: {smiles}.")
        return rxn_mask

    def _sample_rxn(self, mask: np.ndarray = None) -> Tuple[Reaction, int]:
        """Sample a reaction by index."""
        if mask is None:
            irxn_mask = self.IDX_RXNS  #
        else:
            mask =  np.asarray(mask)
            irxn_mask = self.IDX_RXNS[mask]
        idx = self.rng.choice(irxn_mask)
        logger.debug(f"    Sampled reaction with index: {idx} (nreactants: {self.rxns[idx].num_reactant})")
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
            nPossible = len(available_reactants)
            if nPossible==0:
                raise NoReactantAvailableError("Unable to find two reactants for this bimolecular reaction.")
                 # TODO: 2 bi-molecular rxn templates have no matching bblock
            # TODO: use numpy array to avoid type conversion or stick to sampling idx?
            idx = self.rng.choice(nPossible)
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
            canEnd = True
        elif nTrees == 2:
            canExpand = True
            canMerge = True  # TODO: only if rxn is possible
        else:
            raise ValueError

        return np.array((canAdd, canExpand, canMerge, canEnd), dtype=bool)

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
                r1, r2, p, idx_rxn = None, None, None, None
            elif action == "expand":
                for j in range(retries):
                    logger.debug(f"    Try {j}")
                    r1, r2, p, idx_rxn= self._expand(recent_mol)
                    if p is not None: break
                if p is None:
                    # TODO: move to rxn.run_reaction?
                    raise NoReactionPossible(f"Reaction (ID: {idx_rxn}) not possible with: {r1} + {r2}.")

            elif action == "add":
                mol = self._sample_molecule()
                r1, r2, p, idx_rxn = self._expand(mol)
                # Expand this subtree: reactant, reaction, reactant2

            elif action == "merge":
                # merge two subtrees: sample reaction, run it.
                r1, r2 = [node.smiles for node in state]
                # Identify suitable rxn
                # TODO: naive implementation
                rxn_mask1 = self._find_rxn_candidates(r1)
                rxn_mask2 = self._find_rxn_candidates(r2)
                rxn_mask = rxn_mask1 and rxn_mask2
                rxn, idx_rxn = self._sample_rxn(mask=rxn_mask)
                # Run reaction
                p = rxn.run_reaction((r1, r2))
                if p is None:
                    # TODO: move to rxn.run_reaction?
                    raise NoReactionPossible(f"Reaction (ID: {idx_rxn}) not possible with: {r1} + {r2}.")

            # Prepare next iteration
            logger.debug(f"    Ran reaction {r1} + {r2} -> {p}")

            recent_mol = p

            # Update tree
            syntree.update(act, rxn_id=idx_rxn, mol1=r1, mol2=r2, mol_product=p)
            logger.debug(f"SynTree updated.")
            if action == "end":
                break

        logger.debug(f"🙌 SynTree completed.")
        return syntree
