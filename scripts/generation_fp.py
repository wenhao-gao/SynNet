"""
This file contains the code to generate synthetic trees for quary SMILES. 
TODO update description to indicate how this function is different from the rest (to me it seems like a "duplicate" of predict.py and am inclined to delete it but I didn't)
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from sklearn.neighbors import KDTree
from dgllife.model import load_pretrained
from syn_net.utils.data_utils import ReactionSet
from syn_net.utils.predict_utils import synthetic_tree_decoder, mol_fp

import shutup
shutup.please()

# specify which reaction template set to use and the number of bits
rxn_template = 'hb'
nbits = 1024

# define model to use for molecular embedding
model_type = 'gin_supervised_contextpred' # GIN used just for NN search
device = 'cuda:0'
mol_embedder = load_pretrained(model_type).to(device)

# load the purchasable building block embeddings
bb_emb = np.load('/home/whgao/scGen/synth_net/data/enamine_us_emb.npy')


if __name__ == '__main__':

    from syn_net.models.action import Action
    from syn_net.models.reactant1 import Reactant1
    from syn_net.models.rxn_fp import Rxn
    from syn_net.models.reactant2 import Reactant2

    # define path to the reaction templates and purchasable building blocks
    path_to_reaction_file = '/home/whgao/scGen/synth_net/data/reactions_hb.json.gz'
    path_to_building_blocks = '/home/whgao/scGen/synth_net/data/enamine_us_matched.csv.gz'

    # define paths to pretrained modules
    path_to_action = '/home/whgao/scGen/synth_net/synth_net/params/action_net.ckpt'
    path_to_reactant1 = '/home/whgao/scGen/synth_net/synth_net/params/rnt1_net.ckpt'
    path_to_rxn = '/home/whgao/scGen/synth_net/synth_net/params/rxn_net.ckpt'
    path_to_reactant2 = '/home/whgao/scGen/synth_net/synth_net/params/rnt2_net.ckpt'

    # load the purchasable building block SMILES to a dictionary
    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
    bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}

    # load the reaction templates as a ReactionSet object
    rxn_set = ReactionSet()
    rxn_set.load(path_to_reaction_file)
    rxns = rxn_set.rxns
    # with gzip.open(path_reaction_file, 'rb') as f:
    #     rxns = pickle.load(f)

    # load the pre-trained modules
    action_net = Action.load_from_checkpoint(path_to_action)
    reactant1_net = Reactant1.load_from_checkpoint(path_to_reactant1)
    rxn_net = Rxn.load_from_checkpoint(path_to_rxn)
    reactant2_net = Reactant2.load_from_checkpoint(path_to_reactant2)

    action_net.eval()
    reactant1_net.eval()
    rxn_net.eval()
    reactant2_net.eval()

#     query_smis = ['Cc1cc(C)c(S(=O)(=O)Cl)c(C)c1CC(=O)N1CCN(C)c2ccc(-c3ccc(C(CC(=O)NCc4ccc(N(C)C)cc4)N(CCn4c(C)nc5cc(Cl)ccc54)C(=O)NCCCNC(=O)OC(C)(C)C)s3)cc21',
#              'CCN1CC(c2noc(Cn3c(-c4ccc(OCc5coc(-c6ccc(S(C)(=O)=O)cc6)n5)cc4)nc4c5c(c(Br)cc4c3=O)C(=O)c3ccccc3C5=O)n2)C(C(=O)C(C#N)c2ccccc2Br)C1=O',
#               'COc1cccc(CN(CCN(C)C)S(=O)(=O)CC2(F)CC(c3nc(-c4cccc(-c5nnc(CCl)o5)c4)n[nH]3)(C(N(CCCc3nnc(C)n3C)c3cc(Cl)nc(-c4ccoc4)n3)C(F)(F)CCn3ccnn3)C2)c1',
#                'Cc1ccccc1-c1noc(C=C(C2CNC2)C2(C(=O)NC3CCC4CN(C(=O)OC(C)(C)C)CC4C3)CC3(CC(N(C(=O)CN4CCS(=O)(=O)CC4)c4cc(C)c(S(N)(=O)=O)cc4[N+](=O)[O-])CCO3)C2)n1',
#                 'CCOC(=O)C(c1noc(-c2cc(S(=O)(=O)NCc3ccccc3OC)c(Cl)cc2Cl)n1)(C1CCOCC1)C(CCOCC1(C(=O)Oc2c(F)ccc3ccccc23)CCCN(C(=O)OC(C)(C)C)C1)CC(=O)C(C)(C)O',
#                  'COc1ccc(-c2ccccc2C2=Nc3ccc(C)cc3C(=O)N2c2ccc(-c3ccccc3)cc2)c2c(=O)n(-c3ccccn3)c(-c3ccccc3-c3ccc(-c4ccn(-c5cc(F)ccc5OCc5ccccn5)n4)cc3)nc12',
#                   'CCc1nc(CN2CCCC(N(C)C(=O)C(c3nnnn3C3CC4(C3)CC(F)(F)C4)N(C(=O)CC3(CCNC(=O)OC(C)(C)C)COCCO3)c3ccc(S(=O)(=O)N4CCN(C)CC4)cc3[N+](=O)[O-])C2)no1',
#                    'CC(=O)Nc1ccc(NC(=S)N2CCOCC2(Cc2ccccc2)CN(CCc2nc(-c3ccc(-c4ccccc4C(=O)Nc4ccc(-c5ccccc5)cc4)cc3)n[nH]2)S(=O)(=O)CC(F)(F)c2nc(-c3nonc3[N+](=O)[O-])n[nH]2)cc1',
#                     'CC#CCCCCC(O)(Cn1nnnc1CCN(C(=O)NC1CCCC(c2nc(C(=Cc3ccoc3)c3nc(CCNCC(F)(F)F)no3)n[nH]2)C1)c1ccc2c(c1)OCCO2)c1ccc2c(c1)CCC(=O)N2C']
# 
#     query_smis = ['COc1ccc(F)cc1CS(=O)(=O)Nc1ccc(C2CCC(N(C)C(=O)c3ccc(C#N)cc3)CC2)cc1',
#              'COC(=O)c1c(-c2ccccc2)csc1NC(=S)Nc1cc(S(=O)(=O)N2CCCC2)ccc1OCC(F)(F)F',
#               'CCOc1ccc(OCC)c(NC(=S)N(Cc2ccccc2)c2nnc(SCC(=O)NCC3CCCCC3)s2)c1',
#                'COc1cc(OCCC2CCCN2C)ccc1NC(=O)Nc1ccc(I)c(Cl)c1',
#                 'O=C(Nc1cnn2ccc(-c3n~c4cc(C(=O)O)ccc4o3)cc12)OCC1c2ccccc2-c2ccccc21',
#                  'COCC(C)NS(=O)(=O)c1ccc(N(C(C)C)C2CCCN(C(=O)OC(C)(C)C)CC2)c([N+](=O)[O-])c1',
#                   'CCN(CC)CCn1c(-c2cnc(N3CCOC(CNC(=O)OC(C)(C)C)C3)cn2)n~c2cc([N+](=O)[O-])ccc21',
#                    'CCOC(=O)C(=CNc1ccccc1CC)c1noc(-c2ccc(C(=O)Nc3ccccc3NC(=O)OC(C)(C)C)cc2)n1',
#                     'CN(C)S(=O)(=O)c1ccc(CCS(=O)(=O)NCc2cn(-c3ccccc3)nc2-c2ccncc2)cc1']
#
    # define the query SMILES (i.e., molecules to reconstruct)
    query_smis = ['OC1=CC(C(O)CNC)=CC=C1O']

    Trial = len(query_smis)
    num_finish = 0
    num_error = 0
    num_unfinish = 0

    trees = []
    for _ in tqdm(range(Trial)):
        query_smi = query_smis[_]
        print(f'The query SMILES is: {query_smi}')
        z_target = mol_fp(query_smi)
        tree, action = synthetic_tree_decoder(z_target, 
                                              building_blocks,
                                              bb_dict,
                                              rxns, 
                                              mol_embedder, 
                                              action_net, 
                                              reactant1_net, 
                                              rxn_net, 
                                              reactant2_net,
                                              bb_emb=bb_emb,
                                              rxn_template=rxn_template,
                                              n_bits=nbits,
                                              max_step=20)
        if action == 3:
            trees.append(tree)
            num_finish += 1
        elif action == -1:
            num_error += 1
        else:
            num_unfinish += 1
            trees.append(tree)

        tree._print()
        ms = [Chem.MolFromSmiles(sm) for sm in [query_smi, tree.root.smiles]]
        fps = [Chem.RDKFingerprint(x) for x in ms]
        print('Tanimoto similarity is: ', DataStructs.FingerprintSimilarity(fps[0],fps[1]))

    # print some stats
    print('Total trial: ', Trial)
    print('num of finished trees: ', num_finish)
    print('num of unfinished tree: ', num_unfinish)
    print('num of error processes: ', num_error)
