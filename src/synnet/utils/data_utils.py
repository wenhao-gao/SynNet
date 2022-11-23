"""
Here we define the following classes for working with synthetic tree data:
* `Reaction`
* `ReactionSet`
* `NodeChemical`
* `NodeRxn`
* `SyntheticTree`
* `SyntheticTreeSet`
"""
import functools
import gzip
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Set, Tuple, Union

import datamol as dm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from tqdm import tqdm


# the definition of reaction classes below
class Reaction:
    """A chemical reaction defined by a SMARTS pattern."""

    smirks: str
    rxn: Chem.rdChemReactions.ChemicalReaction
    num_reactant: int
    num_agent: int
    num_product: int
    reactant_template: Tuple[str, str]
    product_template: str
    agent_template: str
    available_reactants: Tuple[list[str], Optional[list[str]]]
    rxnname: str
    reference: Any

    def __init__(self, template: str, name: Optional[str] = None, reference: Optional[Any] = None):
        """Initialize a `Reaction`.

        Args:
            template: SMARTS string representing a chemical reaction.
            name: The name of the reaction for downstream analysis.
            reference: (placeholder)
        """
        self.smirks = template.strip()  # SMARTS pattern
        self.name = name
        self.reference = reference

        # Initialize reaction
        self.rxn = dm.reactions.rxn_from_smarts(self.smirks)

        # Extract number of ...
        self.num_reactant = self.rxn.GetNumReactantTemplates()
        if self.num_reactant not in (1, 2):
            raise ValueError("Reaction is neither uni- nor bi-molecular.")
        self.num_agent = self.rxn.GetNumAgentTemplates()
        self.num_product = self.rxn.GetNumProductTemplates()

        # Extract reactants, agents, products
        reactants, agents, products = self.smirks.split(">")

        if self.num_reactant == 1:
            self.reactant_template = list((reactants,))
        elif self.num_reactant == 2:
            self.reactant_template = list(reactants.split("."))

        self.product_template = products
        self.agent_template = agents

    def __repr__(self) -> str:
        return f"Reaction(smarts='{self.smirks}')"

    @classmethod
    def from_dict(cls, attrs: dict):
        """Populate all attributes of the `Reaction` object from a dictionary."""
        rxn = cls(attrs["smirks"])  # only arg without a default
        for k, v in attrs.items():
            rxn.__setattr__(k, v)
        return rxn

    def to_dict(self) -> dict():
        """Returns serializable fields as new dictionary mapping.
        *Excludes* Not-easily-serializable `self.rxn: rdkit.Chem.ChemicalReaction`."""
        import copy

        out = copy.deepcopy(self.__dict__)
        _ = out.pop("rxn")
        return out

    @functools.lru_cache(maxsize=20_000)
    def get_mol(self, smi: Union[str, Chem.Mol]) -> Chem.Mol:
        """Convert smiles to  `RDKit.Chem.Mol`."""
        if isinstance(smi, str):
            return dm.to_mol(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")

    def to_image(self, size: tuple[int, int] = (800, 300)) -> bytes:
        """Returns a png image of the visual represenation for this chemical reaction.

        Usage:
            * In Jupyter:

                >>> from IPython.display import Image
                >>> img = rxn.to_image()
                >>> Image(img)

            * save as image:

                >>> img = rxn.to_image()
                >>> pathlib.Path("out.png").write_bytes(img)

        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(*size)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        image = d2d.GetDrawingText()
        return image

    def is_reactant(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is a reactant of this reaction."""
        mol = self.get_mol(smi)
        return self.rxn.IsMoleculeReactant(mol)

    def is_agent(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is an agent of this reaction."""
        mol = self.get_mol(smi)
        return self.rxn.IsMoleculeAgent(mol)

    def is_product(self, smi):
        """Checks if `smi` is a product of this reaction."""
        mol = self.get_mol(smi)
        return self.rxn.IsMoleculeProduct(mol)

    def is_reactant_first(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` is the first reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` the second reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def run_reaction(
        self,
        reactants: Tuple[Union[str, Chem.Mol, None]],
        keep_main: bool = True,
        allow_to_fail: bool = False,
    ) -> Union[str, None]:
        """Run this reactions with reactants and return corresponding product.

        Args:
            reactants (tuple): Contains SMILES strings for the reactants.
            keep_main (bool): Return main product only or all possibel products. Defaults to True.

        Returns:
            uniqps: SMILES string representing the product or `None` if not reaction possible
        """
        # Input validation.
        if not isinstance(reactants, tuple):
            raise TypeError(f"Unsupported type '{type(reactants)}' for `reactants`.")
        if not len(reactants) in (1, 2):
            raise ValueError(f"Can only run reactions with 1 or 2 reactants, not {len(reactants)}.")

        # Convert all reactants to `Chem.Mol`
        r = tuple(self.get_mol(smiles) for smiles in reactants if smiles is not None)

        # Validate reaction for these reactants
        if self.num_reactant == 1 and len(r) == 2:
            # Provided two reactants for unimolecular reaction -> no rxn possible
            raise AssertionError(f"Provided two reactants ({r=}) for this unimolecular reaction.")
        if self.num_reactant == 1 and not self.is_reactant(r[0]):
            raise AssertionError(
                f"Reactant ({r[0]=}) is not a reactant for this unimolecular reaction."
            )

        if self.num_reactant == 2:
            # Match reactant order with reaction template
            if self.is_reactant_first(r[0]) and self.is_reactant_second(r[1]):
                pass
            elif self.is_reactant_first(r[1]) and self.is_reactant_second(r[0]):
                r = tuple(reversed(r))
            else:  # No reaction possible
                # TODO: Fix: Can happen if both are 1st or 2nd reactant simultanouesly
                raise AssertionError(
                    f"Reactants ({reactants=}) do not match this bimolecular reaction."
                )

        # Run reaction with rdkit magic
        ps = self.rxn.RunReactants(r)

        # Filter for unique products (less magic)
        # Note: Use chain() to flatten the tuple of tuples
        uniqps = list({Chem.MolToSmiles(p) for p in itertools.chain(*ps)})

        # Sanity check
        if not len(uniqps) >= 1:
            # TODO: Raise (custom) exception?
            raise ValueError("Reaction did not yield any products.")

        if keep_main:
            uniqps = uniqps[:1]
        # >>> TODO: Always return list[str] (currently depends on "keep_main")
        uniqps = uniqps[0]
        # <<< ^ delete this line if resolved.

        # Sanity check: Convert SMILES to `Chem.Mol`, then to SMILES again.
        mol = dm.to_mol(uniqps)
        smiles = dm.to_smiles(mol, isomeric=False, allow_to_fail=False)
        if allow_to_fail and smiles is None:
            raise ValueError(f"rdkit.RunReactants() produced invalid product: {uniqps}")
        return smiles

    def _filter_reactants(
        self, smiles: list[str], verbose: bool = False
    ) -> Tuple[list[str], list[str]]:
        """Filters reactants which do not match the reaction."""
        smiles = tqdm(smiles) if verbose else smiles

        if self.num_reactant == 1:  # uni-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            reactants = (reactants_1, [])

        elif self.num_reactant == 2:  # bi-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            reactants_2 = [smi for smi in smiles if self.is_reactant_second(smi)]

            reactants = (reactants_1, reactants_2)

        return reactants

    def set_available_reactants(self, building_blocks: list[str], verbose: bool = False):
        """Finds applicable reactants from a list of building blocks.
        Sets `self.available_reactants`.
        """
        self.available_reactants = self._filter_reactants(building_blocks, verbose=verbose)
        return self

    @property
    def get_available_reactants(self) -> Set[str]:
        return {x for reactants in self.available_reactants for x in reactants}


class ReactionSet:
    """Represents a collection of reactions, for saving and loading purposes."""

    def __init__(self, rxns: Optional[list[Reaction]] = None):
        self.rxns = rxns if rxns is not None else []

    def __repr__(self) -> str:
        return f"ReactionSet ({len(self.rxns)} reactions.)"

    def __len__(self):
        return len(self.rxns)

    def __getitem__(self, index: int):
        if self.rxns is None:
            raise IndexError("No Reactions.")
        return self.rxns[index]

    @classmethod
    def load(cls, file: str):
        """Load a collection of reactions from a `*.json.gz` file."""
        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"

        with gzip.open(file, "r") as f:
            data = json.loads(f.read().decode("utf-8"))

        reactions = [Reaction.from_dict(_rxn) for _rxn in data["reactions"]]
        return cls(reactions)

    def save(self, file: str) -> None:
        """Save a collection of reactions to a `*.json.gz` file."""

        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"

        rxns_as_json = {"reactions": [r.to_dict() for r in self.rxns]}
        with gzip.open(file, "w") as f:
            f.write(json.dumps(rxns_as_json).encode("utf-8"))

    def _print(self, n: int = 3):
        """Debugging-helper method to print `n` reactions as json"""
        for i, r in enumerate(self.rxns):
            if i >= n:
                break
            print(json.dumps(r.to_dict(), indent=2))

    @property
    def num_unimolecular(self) -> int:
        return sum([r.num_reactant == 1 for r in self])

    @property
    def num_bimolecular(self) -> int:
        return sum([r.num_reactant == 2 for r in self])


# the definition of classes for defining synthetic trees below
@dataclass(frozen=True)
class NodeChemical:
    """Represents a chemical node in a synthetic tree.

    Args:
        smiles: Molecule represented as SMILES string.
        parent: Parent molecule represented as SMILES string (i.e. the result of a reaction)
        child: Index of the reaction this object participates in.
        is_leaf: Is this a leaf node in a synthetic tree?
        is_root: Is this a root node in a synthetic tree?
        depth: Depth this node is in tree (+1 for an action, +.5 for a reaction)
        index: Incremental index for all chemical nodes in the tree.
    """

    smiles: Union[str, None] = None
    parent: Union[int, None] = None
    child: Union[int, None] = None
    is_leaf: bool = False
    is_root: bool = False
    depth: float = 0
    index: int = 0


@dataclass(frozen=True)
class NodeRxn:
    """Represents a chemical reaction in a synthetic tree.


    Args:
        rxn_id (None or int): Index corresponding to reaction in a one-hot vector
            of reaction templates.
        rtype (None or int): Indicates if uni- (1) or bi-molecular (2) reaction.
        parent (None or list):
        child (None or list): Contains SMILES strings of reactants which lead to
            the specified reaction.
        depth (float):
        index (int): Indicates the order of this reaction node in the tree.
    """

    rxn_id: Union[int, None] = (None,)
    rtype: Union[int, None] = (None,)
    parent: Union[list, None] = field(default_factory=list)
    child: Union[list, None] = (None,)
    depth: float = (0,)
    index: int = (0,)


class SyntheticTree:
    """Representation of a synthetic tree (syntree).

    Args:
        chemicals (list): A list of chemical nodes, in order of addition.
        reactions (list): A list of reaction nodes, in order of addition.
        actions (list): A list of actions, in order of addition.
        root (NodeChemical): The root node.
        depth (int): Depth of the tree, actions "add" "expand" and "merge" increase depth by 1.
        rxn_id2type (dict): A dictionary that maps reaction indices to reaction
            type (uni- or bi-molecular).
    """

    def __init__(self):
        self.chemicals: list[NodeChemical] = []
        self.reactions: list[NodeRxn] = []
        self.root: Union[NodeChemical, None] = None
        self.depth: float = 0
        self.actions: list[int] = []
        self.rxn_id2type: dict = None

    def __repr__(self) -> str:
        return f"SynTree(depth={self.depth})"

    @classmethod
    def from_dict(cls, attrs: dict):
        """Initialize a `SyntheticTree` from a dictionary."""
        syntree = cls()
        syntree.root = NodeChemical(**attrs["root"])
        syntree.depth = attrs["depth"]
        syntree.actions = attrs["actions"]
        syntree.rxn_id2type = attrs["rxn_id2type"]

        syntree.reactions = [NodeRxn(**_rxn_dict) for _rxn_dict in attrs["reactions"]]
        syntree.chemicals = [NodeChemical(**_chem_dict) for _chem_dict in attrs["chemicals"]]
        return syntree

    def to_dict(self) -> dict:
        """Export this `SyntheticTree` to a dictionary."""
        return {
            "reactions": [r.__dict__ for r in self.reactions],
            "chemicals": [m.__dict__ for m in self.chemicals],
            "root": self.root.__dict__,
            "depth": self.depth,
            "actions": self.actions,
            "rxn_id2type": self.rxn_id2type,
        }

    def _print(self):
        """Print the contents of this `SyntheticTree`."""
        print(f"============SynTree (depth={self.depth:>4.1f})==============")
        print("===============Stored Molecules===============")
        for node in self.chemicals:
            suffix = " (root mol)" if node.is_root else ""
            print(node.smiles, suffix)
        print("===============Stored Reactions===============")
        for node in self.reactions:
            print(f"{node.rxn_id} ({'bi ' if node.rtype==2 else 'uni'})")
        print("===============Followed Actions===============")
        print(self.actions)
        print("==============================================")

    def get_node_index(self, smi: str) -> int:
        """Return the index of the node matching the input SMILES.

        If the query moleucle is not in the tree, return None.
        """
        for node in self.chemicals:
            if smi == node.smiles:
                return node.index
        return None

    def get_state(self) -> list[str]:
        """Get the state of this synthetic tree.
        The most recent root node has 0 as its index.

        Returns:
            state (list): A list contains all root node molecules.
        """
        state = [node.smiles for node in self.chemicals if node.is_root]
        return state[::-1]

    def update(self, action: int, rxn_id: int, mol1: str, mol2: str, mol_product: str):
        """Update this synthetic tree by adding a reaction step.

        Args:
            action (int): Action index, where the indices (0, 1, 2, 3) represent
                (Add, Expand, Merge, and End), respectively.
            rxn_id (int): Index of the reaction occured, where the index can be
               anything in the range [0, len(template_list)-1].
            mol1 (str): SMILES string representing the first reactant.
            mol2 (str): SMILES string representing the second reactant.
            mol_product (str): SMILES string representing the product.
        """
        self.actions.append(int(action))

        if action == 3:  # End
            self.root = self.chemicals[-1]
            self.depth = self.root.depth

        elif action == 2:  # Merge (with bi-mol rxn)
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_mol2 = self.chemicals[self.get_node_index(mol2)]
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=max(node_mol1.depth, node_mol2.depth) + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals),
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False
            node_mol2.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 1 and mol2 is None:  # Expand with uni-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=1,
                parent=None,
                child=[node_mol1.smiles],
                depth=node_mol1.depth + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals),
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 1 and mol2 is not None:  # Expand with bi-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_mol2 = NodeChemical(
                smiles=mol2,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=max(node_mol1.depth, node_mol2.depth) + 0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=node_rxn.depth + 0.5,
                index=len(self.chemicals) + 1,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 0 and mol2 is None:  # Add with uni-mol rxn
            node_mol1 = NodeChemical(
                smiles=mol1,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=1,
                parent=None,
                child=[node_mol1.smiles],
                depth=0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=1,
                index=len(self.chemicals) + 1,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 0 and mol2 is not None:  # Add with bi-mol rxn
            node_mol1 = NodeChemical(
                smiles=mol1,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals),
            )
            node_mol2 = NodeChemical(
                smiles=mol2,
                parent=None,
                child=None,
                is_leaf=True,
                is_root=False,
                depth=0,
                index=len(self.chemicals) + 1,
            )
            node_rxn = NodeRxn(
                rxn_id=rxn_id,
                rtype=2,
                parent=None,
                child=[node_mol1.smiles, node_mol2.smiles],
                depth=0.5,
                index=len(self.reactions),
            )
            node_product = NodeChemical(
                smiles=mol_product,
                parent=None,
                child=node_rxn.rxn_id,
                is_leaf=False,
                is_root=True,
                depth=1,
                index=len(self.chemicals) + 2,
            )

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        else:
            raise ValueError("Check input")
        self.depth = max([node.depth for node in self.reactions]) + 0.5
        return None

    @property
    def chemicals_as_smiles(self) -> list[str]:
        return [node.smiles for node in self.chemicals]

    @property
    def leafs_as_smiles(self) -> list[str]:
        return [node.smiles for node in self.chemicals if node.is_leaf]


class SyntheticTreeSet:
    """Represents a collection of synthetic trees, for saving and loading purposes."""

    sts: list[SyntheticTree]

    def __init__(self, sts: Optional[list[SyntheticTree]] = None):
        self.sts = sts if sts is not None else []

    def __repr__(self) -> str:
        return f"SyntheticTreeSet ({len(self.sts)} syntrees.)"

    def __len__(self):
        return len(self.sts)

    def __getitem__(self, index):
        if self.sts is None:
            raise IndexError("No Synthetic Trees.")
        return self.sts[index]

    @classmethod
    def load(cls, file: str):
        """Load a collection of synthetic trees from a `*.json.gz` file."""
        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"

        with gzip.open(file, "rt") as f:
            data = json.loads(f.read())

        syntrees = [SyntheticTree.from_dict(_syntree) for _syntree in data["trees"]]

        return cls(syntrees)

    def save(self, file: str) -> None:
        """Save a collection of synthetic trees to a `*.json.gz` file."""
        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"

        syntrees_as_json = {"trees": [st.output_dict() for st in self.sts if st is not None]}
        with gzip.open(file, "wt") as f:
            f.write(json.dumps(syntrees_as_json))

    def _print(self, x=3):
        """Helper function for debugging."""
        for i, r in enumerate(self.sts):
            if i >= x:
                break
            print(r.output_dict())


if __name__ == "__main__":
    pass
