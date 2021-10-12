"""
This file contains classes to store and functions to operate on synthetic tree data.
"""
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
import pandas as pd
from tqdm import tqdm
import gzip
import json

"""
The reaction class definition.
"""
class Reaction:
    """
    This class model a chemical reaction based on a SMARTS transformation.

    Args:
        template (str): SMARTS string representing a chemical reaction
        rxnname (str): The name fo the reaction for downstreamed analysis
        smiles: (str): A reaction SMILES string that macthes the SMARTS pattern.
        reference (str): Reference information to the reaction.
    """
    def __init__(self, template=None, rxnname=None, smiles=None, reference=None):
        if template is not None:
            self.smirks = template
            self.rxnname = rxnname
            self.smiles = smiles
            self.reference = reference
            rxn = AllChem.ReactionFromSmarts(self.smirks)
            rdChemReactions.ChemicalReaction.Initialize(rxn)
            self.num_reactant = rxn.GetNumReactantTemplates()
            if self.num_reactant == 0 or self.num_reactant > 2:
                raise ValueError('This reaction is neither uni- nor bi-molecular.')
            self.num_agent = rxn.GetNumAgentTemplates()
            self.num_product = rxn.GetNumProductTemplates()
            if self.num_reactant == 1:
                self.reactant_template = (self.smirks.split('>')[0], )
            else:
                self.reactant_template = (self.smirks.split('>')[0].split('.')[0], self.smirks.split('>')[0].split('.')[1])
            self.product_template = self.smirks.split('>')[2]
            self.agent_template = self.smirks.split('>')[1]

            del rxn
        else:
            self.smirks = None

    def load(self, smirks, num_reactant, num_agent, num_product, reactant_template,
             product_template, agent_template, available_reactants, rxnname, smiles, reference):
        """
        This function load a set of elements and reconstruct a reaction class.
        """
        self.smirks = smirks
        self.num_reactant = num_reactant
        self.num_agent = num_agent
        self.num_product = num_product
        self.reactant_template = reactant_template
        self.product_template = product_template
        self.agent_template = agent_template
        self.available_reactants = available_reactants
        self.rxnname = rxnname
        self.smiles = smiles
        self.reference = reference

    def get_mol(self, smi):
        """
        A internal function that return a RDKit.Chem.Mol object.

        Args:
            smi (str or RDKit.Chem.Mol): the query molecule in either smiles or RDKit.Chem.Mol format.

        Returns:
            Mol (RDKit.Chem.Mol)
        """
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError('The input should be either SMILES or RDKit Mol class.')

    def visualize(self, name='./reaction1_highlight.o.png'):
        """
        A function that plot the chemical translation into a PNG figure.
        One can use "from IPython.display import Image ; Image(name)" to see in a Notebook

        Args:
            name (str): the path to figure

        Returns:
            name (str): the path to figure
        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(800,300)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        png = d2d.GetDrawingText()
        open(name,'wb+').write(png)
        del rxn
        return name

    def is_reactant(self, smi):
        """
        A function that checks if a molecule is the reactant of self reaction

        Args:
            smi (str or RDKit.Chem.Mol): the query molecule in either smiles or RDKit.Chem.Mol format.

        Returns:
            result (boolean): If the molecule is the reactant of self reaction
        """
        rxn = self.get_rxnobj()
        smi = self.get_mol(smi)
        result = rxn.IsMoleculeReactant(smi)
        del rxn
        return result

    def is_agent(self, smi):
        """
        A function that checks if a molecule is a agent of self reaction

        Args:
            smi (str or RDKit.Chem.Mol): the query molecule in either smiles or RDKit.Chem.Mol format.

        Returns:
            result (boolean): If the molecule is a agent of self reaction
        """
        rxn = self.get_rxnobj()
        smi = self.get_mol(smi)
        result = rxn.IsMoleculeAgent(smi)
        del rxn
        return result

    def is_product(self, smi):
        """
        A function that checks if a molecule is the product of self reaction

        Args:
            smi (str or RDKit.Chem.Mol): the query molecule in either smiles or RDKit.Chem.Mol format.

        Returns:
            result (boolean): If the molecule is the product of self reaction
        """
        rxn = self.get_rxnobj()
        smi = self.get_mol(smi)
        result = rxn.IsMoleculeProduct(smi)
        del rxn
        return result

    def is_reactant_first(self, smi):
        """
        A function that checks if a molecule is the first reactant of self reaction, order is determined by the SMARTS pattern

        Args:
            smi (str or RDKit.Chem.Mol): the query molecule in either smiles or RDKit.Chem.Mol format.

        Returns:
            result (boolean): If the molecule is the first reactant of self reaction
        """
        smi = self.get_mol(smi)
        if smi.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(0))):
            return True
        else:
            return False

    def is_reactant_second(self, smi):
        """
        A function that checks if a molecule is the second reactant of self reaction, order is determined by the SMARTS pattern

        Args:
            smi (str or RDKit.Chem.Mol): the query molecule in either smiles or RDKit.Chem.Mol format.

        Returns:
            result (boolean): If the molecule is the second reactant of self reaction
        """
        smi = self.get_mol(smi)
        if smi.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(1))):
            return True
        else:
            return False

    def get_smirks(self):
        """
        A function that returns the SMARTS pattern represents the reaction

        Returns:
            smirks (str): the SMARTS pattern represents the reaction
        """
        return self.smirks

    def get_rxnobj(self):
        """
        A function that returns the Reaction object in RDKit
        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def get_reactant_template(self, ind=0):
        """
        A function that returns the SMARTS pattern represents the reactant

        Args:
            ind (int): the index of the reactant 

        Returns:
            template (str): the SMARTS pattern represents the reactant
        """
        return self.reactant_template[ind]

    def get_product_template(self):
        """
        A function that returns the SMARTS pattern represents the product

        Args:
            ind (int): the index of the reactant 

        Returns:
            template (str): the SMARTS pattern represents the product
        """
        return self.product_template

    def run_reaction(self, reactants, keep_main=True):
        """
        A function that transform the reactants into corresponding product

        Args:
            reactants (str): the index of the reactant 
            keep_main (boolean): if to return main or all possible products

        Returns:
            uniqps (str): the smiles string represents the product
        """

        rxn = self.get_rxnobj()

        if self.num_reactant == 1:

            if isinstance(reactants, (tuple, list)):
                if len(reactants) == 1:
                    r = self.get_mol(reactants[0])
                elif len(reactants) == 2 and reactants[1] is None:
                    r = self.get_mol(reactants[0])
                else:
                    return None

            elif isinstance(reactants, (str, Chem.Mol)):
                r = self.get_mol(reactants)
            else:
                raise TypeError('The input of a uni-molecular reaction should be a SMILES, Chem.Mol or a tuple/list of length 1/2.')

            if not self.is_reactant(r):
                return None

            ps = rxn.RunReactants((r, ))

        elif self.num_reactant == 2:
            if isinstance(reactants, (tuple, list)) and len(reactants) == 2:
                r1 = self.get_mol(reactants[0])
                r2 = self.get_mol(reactants[1])
            else:
                raise TypeError('The input of a bi-molecular reaction should be a tuple/list of length 2.')

            if self.is_reactant_first(r1) and self.is_reactant_second(r2):
                pass
            elif self.is_reactant_first(r2) and self.is_reactant_second(r1):
                r1, r2 = (r2, r1)
            else:
                return None

            ps = rxn.RunReactants((r1, r2))

        else:
            raise ValueError('This reaction is neither uni- nor bi-molecular.')

        uniqps = []
        for p in ps:
            smi = Chem.MolToSmiles(p[0])
            uniqps.append(smi)

        uniqps = list(set(uniqps))

        assert len(uniqps) >= 1

        del rxn

        if keep_main:
            return uniqps[0]
        else:
            return uniqps

    def _filter_reactants(self, smi_list):
        if self.num_reactant == 1:
            smi_w_patt = []
            for smi in tqdm(smi_list):
                if self.is_reactant_first(smi):
                    smi_w_patt.append(smi)
            return (smi_w_patt, )
        elif self.num_reactant == 2:
            smi_w_patt1 = []
            smi_w_patt2 = []
            for smi in tqdm(smi_list):
                if self.is_reactant_first(smi):
                    smi_w_patt1.append(smi)
                if self.is_reactant_second(smi):
                    smi_w_patt2.append(smi)
            return (smi_w_patt1, smi_w_patt2)
        else:
            raise ValueError('This reaction is neither uni- nor bi-molecular.')

    def set_available_reactants(self, building_block_list):
        """
        A function that finds the applicable building block from a list of purchasable building blocks

        Args:
            building_block_list (list): the list of purchasable building blocks
        """
        self.available_reactants = self._filter_reactants(building_block_list)
        return None


class ReactionSet:
    """
    A class represent a set of reactions, for saving and loading purposes

    Arritbute:
        rxns (list): a list of Reaction object, one can initialize the class witha list or None
    """
    def __init__(self, rxns=None):
        if rxns is None:
            self.rxns = []
        else:
            self.rxns = rxns

    def load(self, json_file):
        """
        A function that load json formatted reaction file.
        
        Args:
            json_file (str): the path to the stored reaction file.
        """

        with gzip.open(json_file, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))

        for r_dict in data['reactions']:
            r = Reaction()
            r.load(**r_dict)
            self.rxns.append(r)

    def save(self, json_file):
        """
        A function that save the reaction set to a json formatted file.
        
        Args:
            json_file (str): the path to the stored reaction file.
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


"""
The definition of classes for synthetic trees
"""
class NodeChemical:
    """
    A class represent a chemical node in a synthetic tree
    """
    def __init__(self, smiles=None, parent=None, child=None, is_leaf=False, is_root=False, depth=0, index=0):
        self.smiles = smiles
        self.parent = parent
        self.child = child
        self.is_leaf = is_leaf
        self.is_root = is_root
        self.depth = depth
        self.index = index


class NodeRxn:
    """
    A class represent a reaction node in a synthetic tree
    """
    def __init__(self, rxn_id=None, rtype=None, parent=[], child=None, depth=0, index=0):
        self.rxn_id = rxn_id
        self.parent = parent
        self.child = child
        self.depth = depth
        self.index = index
        self.rtype = rtype


class SyntheticTree:
    """
    A class represent a reaction node in a synthetic tree

    Args:
        chemicals (list): A list of chemical node in the order of addition.
        reactions (list): A list of reaction node in the order of addition.
        actions (list): A list of actions in the order of addition.
        root: (NodeChemical): The root node.
        depth (int): The depth of the tree.
        rxn_id2type (dict): A dictionary that maps reaction indices to reaction type (uni- or bi- molecular)
    """
    def __init__(self, tree=None):
        self.chemicals = []
        self.reactions = []
        self.root = None
        self.depth = 0
        self.actions = []
        self.rxn_id2type = None

        if tree is not None:
            self.read(tree)

    def read(self, data):
        """
        A function that load diction formatted synthetic tree data.
        
        Args:
            data (dict): A dict represent a synthetic tree
        """

        # with gzip.open(json_file, 'r') as f:
        #     data = json.loads(f.read().decode('utf-8'))

        self.root = NodeChemical(**data['root'])
        self.depth = data['depth']
        self.actions = data['actions']
        self.rxn_id2type = data['rxn_id2type']

        for r_dict in data['reactions']:
            r = NodeRxn(**r_dict)
            self.reactions.append(r)

        for m_dict in data['chemicals']:
            r = NodeChemical(**m_dict)
            self.chemicals.append(r)

    def output_dict(self):
        """
        A function that export diction formatted synthetic tree data.
        
        Returns:
            data (dict): A dict represent a synthetic tree
        """
        return {'reactions': [r.__dict__ for r in self.reactions],
                'chemicals': [m.__dict__ for m in self.chemicals],
                'root': self.root.__dict__,
                'depth': self.depth,
                'actions': self.actions,
                'rxn_id2type': self.rxn_id2type}

        # with gzip.open(json_file, 'w') as f:
        #     f.write(json.dumps(self_dict).encode('utf-8'))

    def _print(self):
        """
        A function that print the content in the synthetic tree
        
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
        Return state with order, the most recent root node has 0 as its index

        Args:
            smi (str): A SMILES string that represents the query molecule
        
        Returns:
            index (int): index of chemical node corresponding to the query moleucle. If the query moleucle is 
                         not in tree, return None.
        """
        for node in self.chemicals:
            if smi == node.smiles:
                return node.index
        return None

    def get_state(self):
        """
        Return state with order, the most recent root node has 0 as its index

        Returns:
            state (list): A list contains all root node molecules
        """
        state = []
        for mol in self.chemicals:
            if mol.is_root:
                state.append(mol.smiles)
        return state[::-1]

    def update(self, action, rxn_id, mol1, mol2, mol_product):
        """
        A function that update a reaction step to self tree

        Args:
            action (int): action belongs to (0, 1, 2, 3) represents (Add, Expand, Merge and End)
            rxn_id (int): index of reaction occured, belongs to [0, len(template_list)-1]
            mol1 (str): smiles string represent the first reactant
            mol2 (str): smiles string represent the second reactant
            mol_product (str): smiles string represent the product
        """
        self.actions.append(int(action))

        if action == 3:
            # End
            self.root = self.chemicals[-1]
            self.depth = self.root.depth

        elif action == 2:
            # Merge with bi-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_mol2 = self.chemicals[self.get_node_index(mol2)]
            node_rxn = NodeRxn(rxn_id, 2, parent=None, child=[node_mol1.smiles, node_mol2.smiles], depth=max(node_mol1.depth, node_mol2.depth)+0.5, index=len(self.reactions))
            node_product = NodeChemical(mol_product, parent=None, child=node_rxn.rxn_id, is_leaf=False, is_root=True, depth=node_rxn.depth+0.5, index=len(self.chemicals))

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False
            node_mol2.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 1 and mol2 is None:
            # Expand with uni-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_rxn = NodeRxn(rxn_id, 1, parent=None, child=[node_mol1.smiles], depth=node_mol1.depth+0.5, index=len(self.reactions))
            node_product = NodeChemical(mol_product, parent=None, child=node_rxn.rxn_id, is_leaf=False, is_root=True, depth=node_rxn.depth+0.5, index=len(self.chemicals))

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 1 and mol2 is not None:
            # Expand with bi-mol rxn
            node_mol1 = self.chemicals[self.get_node_index(mol1)]
            node_mol2 = NodeChemical(mol2, parent=None, child=None, is_leaf=True, is_root=False, depth=0, index=len(self.chemicals))
            node_rxn = NodeRxn(rxn_id, 2, parent=None, child=[node_mol1.smiles, node_mol2.smiles], depth=max(node_mol1.depth, node_mol2.depth)+0.5, index=len(self.reactions))
            node_product = NodeChemical(mol_product, parent=None, child=node_rxn.rxn_id, is_leaf=False, is_root=True, depth=node_rxn.depth+0.5, index=len(self.chemicals)+1)

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id
            node_mol2.parent = node_rxn.rxn_id
            node_mol1.is_root = False

            self.chemicals.append(node_mol2)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 0 and mol2 is None:
            # Add with uni-mol rxn
            node_mol1 = NodeChemical(mol1, parent=None, child=None, is_leaf=True, is_root=False, depth=0, index=len(self.chemicals))
            node_rxn = NodeRxn(rxn_id, 1, parent=None, child=[node_mol1.smiles], depth=0.5, index=len(self.reactions))
            node_product = NodeChemical(mol_product, parent=None, child=node_rxn.rxn_id, is_leaf=False, is_root=True, depth=1, index=len(self.chemicals)+1)

            node_rxn.parent = node_product.smiles
            node_mol1.parent = node_rxn.rxn_id

            self.chemicals.append(node_mol1)
            self.chemicals.append(node_product)
            self.reactions.append(node_rxn)

        elif action == 0 and mol2 is not None:
            # Add with bi-mol rxn
            node_mol1 = NodeChemical(mol1, parent=None, child=None, is_leaf=True, is_root=False, depth=0, index=len(self.chemicals))
            node_mol2 = NodeChemical(mol2, parent=None, child=None, is_leaf=True, is_root=False, depth=0, index=len(self.chemicals)+1)
            node_rxn = NodeRxn(rxn_id, 2, parent=None, child=[node_mol1.smiles, node_mol2.smiles], depth=0.5, index=len(self.reactions))
            node_product = NodeChemical(mol_product, parent=None, child=node_rxn.rxn_id, is_leaf=False, is_root=True, depth=1, index=len(self.chemicals)+2)

            node_rxn.parent = node_product.smiles
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
    A class represent a list of synthetic trees, for saving and loading purposes

    Arritbute:
        sts (list): a list of synthetic trees, one can initialize the class with a list or None
    """
    def __init__(self, sts=None):
        if sts is None:
            self.sts = []
        else:
            self.sts = sts

    def load(self, json_file):
        """
        A function that load json formatted synthetic tree file.
        
        Args:
            json_file (str): the path to the stored synthetic tree file.
        """

        with gzip.open(json_file, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))

        for st_dict in data['trees']:
            if st_dict is None:
                self.sts.append(None)
            else:
                st = SyntheticTree(st_dict)
                self.sts.append(st)

    def save(self, json_file):
        """
        A function that save the synthetic tree set to a json formatted file.
        
        Args:
            json_file (str): the path to the stored synthetic tree file.
        """
        st_list = {'trees': [st.output_dict() if st is not None else None for st in self.sts]}
        with gzip.open(json_file, 'w') as f:
            f.write(json.dumps(st_list).encode('utf-8'))

    def __len__(self):
        return len(self.sts)

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
