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

# TODO add docstrings, types, comments (not much description in this file)


"""
The reaction class definition.
"""
class Reaction:
    """
    TODO
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
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError('The input should be either SMILES or RDKit Mol class.')

    def visualize(self, name='./reaction1_highlight.o.png'):
        # from IPython.display import Image ; Image(name) to see in iPython
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(800,300)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        png = d2d.GetDrawingText()
        open(name,'wb+').write(png)
        del rxn
        return name

    def is_reactant(self, smi):
        rxn = self.get_rxnobj()
        smi = self.get_mol(smi)
        result = rxn.IsMoleculeReactant(smi)
        del rxn
        return result

    def is_agent(self, smi):
        rxn = self.get_rxnobj()
        smi = self.get_mol(smi)
        result = rxn.IsMoleculeAgent(smi)
        del rxn
        return result

    def is_product(self, smi):
        rxn = self.get_rxnobj()
        smi = self.get_mol(smi)
        result = rxn.IsMoleculeProduct(smi)
        del rxn
        return result

    def is_reactant_first(self, smi):
        smi = self.get_mol(smi)
        if smi.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(0))):
            return True
        else:
            return False

    def is_reactant_second(self, smi):
        smi = self.get_mol(smi)
        if smi.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(1))):
            return True
        else:
            return False

    def get_smirks(self):
        return self.smirks

    def get_rxnobj(self):
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def get_reactant_template(self, ind=0):
        return self.reactant_template[ind]

    def get_product_template(self):
        return self.product_template

    def run_reaction(self, reactants, keep_main=True):

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
        self.available_reactants = self._filter_reactants(building_block_list)
        return None


class ReactionSet:
    """
    TODO
    """
    def __init__(self, rxns=None):
        if rxns is None:
            self.rxns = []
        else:
            self.rxns = rxns

    def load(self, json_file):

        with gzip.open(json_file, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))

        for r_dict in data['reactions']:
            r = Reaction()
            r.load(**r_dict)
            self.rxns.append(r)

    def save(self, json_file):
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
    TODO
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
    TODO
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
    TODO
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
        return {'reactions': [r.__dict__ for r in self.reactions],
                'chemicals': [m.__dict__ for m in self.chemicals],
                'root': self.root.__dict__,
                'depth': self.depth,
                'actions': self.actions,
                'rxn_id2type': self.rxn_id2type}

        # with gzip.open(json_file, 'w') as f:
        #     f.write(json.dumps(self_dict).encode('utf-8'))

    def _print(self):
        print('===============Stored Molecules===============')
        for node in self.chemicals:
            print(node.smiles, node.is_root)
        print('===============Stored Reactions===============')
        for node in self.reactions:
            print(node.rxn_id, node.rtype)
        print('===============Followed Actions===============')
        print(self.actions)

    def get_node_index(self, smi):
        for node in self.chemicals:
            if smi == node.smiles:
                return node.index
        return None

    def get_state(self):
        """
        Return state with order, the most recent root node has 0 as its index
        """
        state = []
        for mol in self.chemicals:
            if mol.is_root:
                state.append(mol.smiles)
        return state[::-1]

    def update(self, action, rxn_id, mol1, mol2, mol_product):
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
    TODO
    """
    def __init__(self, sts=None):
        if sts is None:
            self.sts = []
        else:
            self.sts = sts

    def load(self, json_file):

        with gzip.open(json_file, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))

        for st_dict in data['trees']:
            if st_dict is None:
                self.sts.append(None)
            else:
                st = SyntheticTree(st_dict)
                self.sts.append(st)

    def save(self, json_file):
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
