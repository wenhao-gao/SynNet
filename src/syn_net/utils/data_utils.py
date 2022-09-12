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
import itertools
import gzip
import json
from typing import Any, Optional, Tuple, Union, Set

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from tqdm import tqdm


# the definition of reaction classes below
class Reaction:
    """
    This class models a chemical reaction based on a SMARTS transformation.

    Args:
        template (str): SMARTS string representing a chemical reaction.
        rxnname (str): The name of the reaction for downstream analysis.
        smiles: (str): A reaction SMILES string that macthes the SMARTS pattern.
        reference (str): Reference information for the reaction.
    """
    smirks: str # SMARTS pattern
    rxn: Chem.rdChemReactions.ChemicalReaction
    num_reactant: int
    num_agent: int
    num_product: int
    reactant_template: Tuple[str,str]
    product_template: str
    agent_templat: str
    available_reactants: Tuple[list[str],Optional[list[str]]]
    rxnname: str
    smiles: Any
    reference: Any

    def __init__(self, template=None, rxnname=None, smiles=None, reference=None):

        if template is not None:
            # define a few attributes based on the input
            self.smirks    = template
            self.rxnname   = rxnname
            self.smiles    = smiles
            self.reference = reference

            # compute a few additional attributes
            self.rxn = self.__init_reaction(self.smirks)

            # Extract number of ...
            self.num_reactant = self.rxn.GetNumReactantTemplates()
            if self.num_reactant not in (1,2):
                raise ValueError('Reaction is neither uni- nor bi-molecular.')
            self.num_agent = self.rxn.GetNumAgentTemplates()
            self.num_product = self.rxn.GetNumProductTemplates()

            # Extract reactants, agents, products
            reactants, agents, products = self.smirks.split(">")

            if self.num_reactant == 1:
                self.reactant_template = list((reactants, ))
            else:
                self.reactant_template = list(reactants.split("."))
            self.product_template = products
            self.agent_template = agents
        else:
            self.smirks = None

    def __init_reaction(self,smirks: str) -> Chem.rdChemReactions.ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = AllChem.ReactionFromSmarts(smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def load(self, smirks, num_reactant, num_agent, num_product, reactant_template,
             product_template, agent_template, available_reactants, rxnname, smiles, reference):
        """
        This function loads a set of elements and reconstructs a `Reaction` object.
        """
        self.smirks              = smirks
        self.num_reactant        = num_reactant
        self.num_agent           = num_agent
        self.num_product         = num_product
        self.reactant_template   = list(reactant_template)
        self.product_template    = product_template
        self.agent_template      = agent_template
        self.available_reactants = list(available_reactants) # TODO: use Tuple[list,list] here
        self.rxnname             = rxnname
        self.smiles              = smiles
        self.reference           = reference

    @functools.lru_cache(maxsize=20)
    def get_mol(self, smi: Union[str,Chem.Mol]) -> Chem.Mol:
        """
        A internal function that returns an `RDKit.Chem.Mol` object.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            RDKit.Chem.Mol
        """
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")


    def visualize(self, name='./reaction1_highlight.o.png'):
        """
        A function that plots the chemical translation into a PNG figure.
        One can use "from IPython.display import Image ; Image(name)" to see it
        in a Python notebook.

        Args:
            name (str): The path to the figure.

        Returns:
            name (str): The path to the figure.
        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(800,300)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        png = d2d.GetDrawingText()
        open(name,'wb+').write(png)
        del rxn
        return name

    def is_reactant(self, smi: Union[str,Chem.Mol]) -> bool:
        """Checks if `smi` is a reactant of this reaction."""
        smi    = self.get_mol(smi)
        return self.rxn.IsMoleculeReactant(smi)

    def is_agent(self, smi: Union[str,Chem.Mol]) -> bool:
        """Checks if `smi` is an agent of this reaction."""
        smi    = self.get_mol(smi)
        return self.rxn.IsMoleculeAgent(smi)

    def is_product(self, smi):
        """Checks if `smi` is a product of this reaction."""
        smi    = self.get_mol(smi)
        return self.rxn.IsMoleculeProduct(smi)

    def is_reactant_first(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` is the first reactant in this reaction """
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, smi: Union[str,Chem.Mol]) -> bool:
        """Check if `smi` the second reactant in this reaction """
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def run_reaction(self, reactants: Tuple[Union[str,Chem.Mol,None]], keep_main: bool=True) -> Union[str,None]:
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
        if not len(reactants) in (1,2):
            raise ValueError(f"Can only run reactions with 1 or 2 reactants, not {len(reactants)}.")

        rxn = self.rxn  # TODO: investigate if this is necessary (if not, delete "delete rxn below")

        # Convert all reactants to `Chem.Mol`
        r: Tuple = tuple(self.get_mol(smiles) for smiles in reactants if smiles is not None)


        if self.num_reactant == 1:
            if not self.is_reactant(r[0]):
                return None
        elif self.num_reactant == 2:
            # Match reactant order with reaction template
            if self.is_reactant_first(r[0]) and self.is_reactant_second(r[1]):
                pass
            elif self.is_reactant_first(r[1]) and self.is_reactant_second(r[0]):
                r = tuple(reversed(r))
            else: # No reaction possible
                return None
        else:
            raise ValueError('This reaction is neither uni- nor bi-molecular.')
        # Run reaction with rdkit magic
        ps = rxn.RunReactants(r)

        # Filter for unique products (less magic)
        # Note: Use chain() to flatten the tuple of tuples
        uniqps = list({Chem.MolToSmiles(p) for p in itertools.chain(*ps)})

        # Sanity check
        if not len(uniqps) >= 1:
            # TODO: Raise (custom) exception?
            raise ValueError("Reaction did not yield any products.")

        del rxn

        if keep_main:
            uniqps = uniqps[:1]
        # >>> TODO: Always return list[str] (currently depends on "keep_main")
        uniqps = uniqps[0]
        # <<< ^ delete this line if resolved.
        return uniqps

    def _filter_reactants(self, smiles: list[str],verbose: bool=False) -> Tuple[list[str],list[str]]:
        """
        Filters reactants which do not match the reaction.

        Args:
            smiles: Possible reactants for this reaction.

        Returns:
            :lists of SMILES which match either the first
                reactant, or, if applicable, the second reactant.

        Raises:
            ValueError: If `self` is not a uni- or bi-molecular reaction.
        """
        smiles = tqdm(smiles) if verbose else smiles

        if self.num_reactant == 1:  # uni-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            return (reactants_1, )

        elif self.num_reactant == 2:  # bi-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            reactants_2 = [smi for smi in smiles if self.is_reactant_second(smi)]

            return (reactants_1, reactants_2)
        else:
            raise ValueError('This reaction is neither uni- nor bi-molecular.')

    def set_available_reactants(self, building_blocks: list[str],verbose: bool=False):
        """
        Finds applicable reactants from a list of building blocks.
        Sets `self.available_reactants`.

        Args:
            building_blocks: Building blocks as SMILES strings.
        """
        self.available_reactants = self._filter_reactants(building_blocks,verbose=verbose)
        return self

    @property
    def get_available_reactants(self) -> Set[str]:
        return {x for reactants in self.available_reactants for x in reactants}



class ReactionSet:
    """
    A class representing a set of reactions, for saving and loading purposes.

    Arritbutes:
        rxns (list or None): Contains `Reaction` objects. One can initialize the
            class with a list or None object, the latter of which is used to
            define an empty list.
    """
    def __init__(self, rxns=None):
        if rxns is None:
            self.rxns = []
        else:
            self.rxns = rxns

    def load(self, json_file):
        """
        A function that loads reactions from a JSON-formatted file.

        Args:
            json_file (str): The path to the stored reaction file.
        """

        with gzip.open(json_file, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))

        for r_dict in data['reactions']:
            r = Reaction()
            r.load(**r_dict)
            self.rxns.append(r)
        return self

    def save(self, json_file):
        """
        A function that saves the reaction set to a JSON-formatted file.

        Args:
            json_file (str): The path to the stored reaction file.
        """
        r_list = {'reactions': [r.__dict__ for r in self.rxns]}
        with gzip.open(json_file, 'w') as f:
            f.write(json.dumps(r_list).encode('utf-8'))

    def __len__(self):
        return len(self.rxns)

    def _print(self, x=3):
        # For debugging
        for i, r in enumerate(self.rxns):
            if i >= x:
                break
            print(r.__dict__)


# the definition of classes for defining synthetic trees below
class NodeChemical:
    """
    A class representing a chemical node in a synthetic tree.

    Args:
        smiles (None or str): SMILES string representing molecule.
        parent (None or int):
        child (None or int): Indicates reaction which molecule participates in.
        is_leaf (bool): Indicates if this is a leaf node.
        is_root (bool): Indicates if this is a root node.
        depth (float):
        index (int): Indicates the order of this chemical node in the tree.
    """
    def __init__(self, smiles=None, parent=None, child=None, is_leaf=False,
                 is_root=False, depth=0, index=0):
        self.smiles  = smiles
        self.parent  = parent
        self.child   = child
        self.is_leaf = is_leaf
        self.is_root = is_root
        self.depth   = depth
        self.index   = index


class NodeRxn:
    """
    A class representing a reaction node in a synthetic tree.

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
    def __init__(self, rxn_id=None, rtype=None, parent=[],
                 child=None, depth=0, index=0):
        self.rxn_id = rxn_id
        self.rtype  = rtype
        self.parent = parent
        self.child  = child
        self.depth  = depth
        self.index  = index


class SyntheticTree:
    """
    A class representing a synthetic tree.

    Args:
        chemicals (list): A list of chemical nodes, in order of addition.
        reactions (list): A list of reaction nodes, in order of addition.
        actions (list): A list of actions, in order of addition.
        root (NodeChemical): The root node.
        depth (int): The depth of the tree.
        rxn_id2type (dict): A dictionary that maps reaction indices to reaction
            type (uni- or bi-molecular).
    """
    def __init__(self, tree=None):
        self.chemicals: list[NodeChemical]   = []
        self.reactions:list [Reaction]   = []
        self.root        = None
        self.depth: float= 0
        self.actions     = []
        self.rxn_id2type = None

        if tree is not None:
            self.read(tree)

    def read(self, data):
        """
        A function that loads a dictionary from synthetic tree data.

        Args:
            data (dict): A dictionary representing a synthetic tree.
        """
        self.root        = NodeChemical(**data['root'])
        self.depth       = data['depth']
        self.actions     = data['actions']
        self.rxn_id2type = data['rxn_id2type']

        for r_dict in data['reactions']:
            r = NodeRxn(**r_dict)
            self.reactions.append(r)

        for m_dict in data['chemicals']:
            r = NodeChemical(**m_dict)
            self.chemicals.append(r)

    def output_dict(self):
        """
        A function that exports dictionary-formatted synthetic tree data.

        Returns:
            data (dict): A dictionary representing a synthetic tree.
        """
        return {'reactions': [r.__dict__ for r in self.reactions],
                'chemicals': [m.__dict__ for m in self.chemicals],
                'root': self.root.__dict__,
                'depth': self.depth,
                'actions': self.actions,
                'rxn_id2type': self.rxn_id2type}

    def _print(self):
        """
        A function that prints the contents of the synthetic tree.
        """
        print('===============Stored Molecules===============')
        for node in self.chemicals:
            print(node.smiles, node.is_root)
        print('===============Stored Reactions===============')
        for node in self.reactions:
            print(node.rxn_id, node.rtype)
        print('===============Followed Actions===============')
        print(self.actions)

    def get_node_index(self, smi):
        """
        Returns the index of the node matching the input SMILES.

        Args:
            smi (str): A SMILES string that represents the query molecule.

        Returns:
            index (int): Index of chemical node corresponding to the query
                molecule. If the query moleucle is not in the tree, return None.
        """
        for node in self.chemicals:
            if smi == node.smiles:
                return node.index
        return None

    def get_state(self) -> list[NodeChemical]:
        """Get the state of this synthetic tree.
        The most recent root node has 0 as its index.

        Returns:
            state (list): A list contains all root node molecules.
        """
        state = [mol for mol in self.chemicals if mol.is_root]
        return state[::-1]

    def update(self, action: int, rxn_id:int, mol1: str, mol2: str, mol_product:str):
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

        if action == 3: # End
            self.root = self.chemicals[-1]
            self.depth = self.root.depth

        elif action == 2: # Merge (with bi-mol rxn)
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_mol2 = self.chemicals[self.get_node_index(mol2)]
            node_rxn = NodeRxn(rxn_id=rxn_id,
                               rtype=2,
                               parent=None,
                               child=[node_mol1.smiles, node_mol2.smiles],
                               depth=max(node_mol1.depth, node_mol2.depth)+0.5,
                               index=len(self.reactions))
            node_product = NodeChemical(smiles=mol_product,
                                        parent=None,
                                        child=node_rxn.rxn_id,
                                        is_leaf=False,
                                        is_root=True,
                                        depth=node_rxn.depth+0.5,
                                        index=len(self.chemicals))

            node_rxn.parent   = node_product.smiles
            node_mol1.parent  = node_rxn.rxn_id
            node_mol2.parent  = node_rxn.rxn_id
            node_mol1.is_root = False
            node_mol2.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 1 and mol2 is None: # Expand with uni-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_rxn = NodeRxn(rxn_id=rxn_id,
                               rtype=1,
                               parent=None,
                               child=[node_mol1.smiles],
                               depth=node_mol1.depth+0.5,
                               index=len(self.reactions))
            node_product = NodeChemical(smiles=mol_product,
                                        parent=None,
                                        child=node_rxn.rxn_id,
                                        is_leaf=False,
                                        is_root=True,
                                        depth=node_rxn.depth+0.5,
                                        index=len(self.chemicals))

            node_rxn.parent   = node_product.smiles
            node_mol1.parent  = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 1 and mol2 is not None: # Expand with bi-mol rxn
            node_mol1    = self.chemicals[self.get_node_index(mol1)]
            node_mol2    = NodeChemical(smiles=mol2,
                                        parent=None,
                                        child=None,
                                        is_leaf=True,
                                        is_root=False,
                                        depth=0,
                                        index=len(self.chemicals))
            node_rxn     = NodeRxn(rxn_id=rxn_id,
                                   rtype=2,
                                   parent=None,
                                   child=[node_mol1.smiles,
                                   node_mol2.smiles],
                                   depth=max(node_mol1.depth, node_mol2.depth)+0.5,
                                   index=len(self.reactions))
            node_product = NodeChemical(smiles=mol_product,
                                        parent=None,
                                        child=node_rxn.rxn_id,
                                        is_leaf=False,
                                        is_root=True,
                                        depth=node_rxn.depth+0.5,
                                        index=len(self.chemicals)+1)

            node_rxn.parent   = node_product.smiles
            node_mol1.parent  = node_rxn.rxn_id
            node_mol2.parent  = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 0 and mol2 is None: # Add with uni-mol rxn
            node_mol1    = NodeChemical(smiles=mol1,
                                        parent=None,
                                        child=None,
                                        is_leaf=True,
                                        is_root=False,
                                        depth=0,
                                        index=len(self.chemicals))
            node_rxn     = NodeRxn(rxn_id=rxn_id,
                                   rtype=1,
                                   parent=None,
                                   child=[node_mol1.smiles],
                                   depth=0.5,
                                   index=len(self.reactions))
            node_product = NodeChemical(smiles=mol_product,
                                        parent=None,
                                        child=node_rxn.rxn_id,
                                        is_leaf=False,
                                        is_root=True,
                                        depth=1,
                                        index=len(self.chemicals)+1)

            node_rxn.parent  = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 0 and mol2 is not None: # Add with bi-mol rxn
            node_mol1    = NodeChemical(smiles=mol1,
                                        parent=None,
                                        child=None,
                                        is_leaf=True,
                                        is_root=False,
                                        depth=0,
                                        index=len(self.chemicals))
            node_mol2    = NodeChemical(smiles=mol2,
                                        parent=None,
                                        child=None,
                                        is_leaf=True,
                                        is_root=False,
                                        depth=0,
                                        index=len(self.chemicals)+1)
            node_rxn     = NodeRxn(rxn_id=rxn_id,
                                   rtype=2,
                                   parent=None,
                                   child=[node_mol1.smiles, node_mol2.smiles],
                                   depth=0.5,
                                   index=len(self.reactions))
            node_product = NodeChemical(smiles=mol_product,
                                        parent=None,
                                        child=node_rxn.rxn_id,
                                        is_leaf=False,
                                        is_root=True,
                                        depth=1,
                                        index=len(self.chemicals)+2)

            node_rxn.parent  = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        else:
            raise ValueError('Check input')

        return None


class SyntheticTreeSet:
    """
    A class representing a set of synthetic trees, for saving and loading purposes.

    Arritbute:
        sts (list): Contains `SyntheticTree`s. One can initialize the class with
            either a list of synthetic trees or None, in which case an empty
            list is created.
    """
    def __init__(self, sts=None):
        if sts is None:
            self.sts = []
        else:
            self.sts = sts

    def __len__(self):
        return len(self.sts)

    def __getitem__(self,index):
        if self.sts is None: raise IndexError("No Synthetic Trees.")
        return self.sts[index]

    def load(self, json_file):
        """
        A function that loads a JSON-formatted synthetic tree file.

        Args:
            json_file (str): The path to the stored synthetic tree file.
        """
        with gzip.open(json_file, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))

        for st_dict in data['trees']:
            if st_dict is None:
                self.sts.append(None)
            else:
                st = SyntheticTree(st_dict)
                self.sts.append(st)
        return self

    def save(self, json_file):
        """
        A function that saves the synthetic tree set to a JSON-formatted file.

        Args:
            json_file (str): The path to the stored synthetic tree file.
        """
        st_list = {
            'trees': [st.output_dict() if st is not None else None for st in self.sts]
        }
        with gzip.open(json_file, 'w') as f:
            f.write(json.dumps(st_list).encode('utf-8'))

    def _print(self, x=3):
        # For debugging
        for i, r in enumerate(self.sts):
            if i >= x:
                break
            print(r.output_dict())


if __name__ == '__main__':
    """
    A test run to find available reactants for a set of reaction templates.
    """
    path_to_building_blocks = '/home/whgao/shared/Data/scGen/enamine_5k.csv.gz'
    # path_to_rxn_templates = '/home/whgao/shared/Data/scGen/rxn_set_hartenfeller.txt'
    path_to_rxn_templates = '/home/whgao/shared/Data/scGen/rxn_set_pis_test.txt'

    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
    rxns = []
    for line in open(path_to_rxn_templates, 'rt'):
        rxn = Reaction(line.split('|')[1].strip())
        rxn.set_available_reactants(building_blocks)
        rxns.append(rxn)

    r = ReactionSet(rxns)
    r.save('reactions_pis_test.json.gz')
