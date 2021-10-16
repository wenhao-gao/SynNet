"""
Unit tests for the data preparation.
"""
import unittest
import os
import dill as pickle
import pandas as pd
import gzip
from tqdm import tqdm
from scipy import sparse
import numpy as np
from syn_net.utils.prep_utils import organize, synthetic_tree_generator
from syn_net.utils.data_utils import SyntheticTreeSet, Reaction, ReactionSet

class TestDataPrep(unittest.TestCase):
    """
    Tests for the data preparation: (1) data splitting, (2) featurization, (3) training
    data preparation for each network.
    """
    def test_process_rxn_templates(self):
        """
        Tests the rxn templates processing.
        """
        # the following file contains the three templates at the top of
        # 'SynNet/data/rxn_set_hb.txt'
        path_to_rxn_templates = './data/rxn_set_hb_test.txt'

        # load the reference building blocks (1K here)
        path_to_building_blocks = './data/building_blocks_matched.csv.gz'
        building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()

        # load the reaction templates
        rxn_templates = []
        with open(path_to_rxn_templates, 'rt') as rxn_template_file:
            for line in rxn_template_file:
                rxn = Reaction(line.split('|')[1].strip())
                rxn.set_available_reactants(building_block_list=building_blocks)
                rxn_templates.append(rxn)

        # save the templates as a ReactionSet
        r = ReactionSet(rxn_templates)
        r.save('./data/rxns_hb.json.gz')

        # load the reference reaction templates
        path_to_ref_rxn_templates = './data/ref/rxns_hb.json.gz'
        r_ref = ReactionSet()
        r_ref.load(path_to_ref_rxn_templates)
        
        # check here that the templates were correctly saved as a ReactionSet by
        # comparing to a provided reference file in 'SynNet/tests/data/ref/'
        for rxn_idx, rxn in enumerate(r.rxns):
            rxn = rxn.__dict__
            ref_rxn = r_ref.rxns[rxn_idx].__dict__
            self.assertTrue(rxn == ref_rxn)
    
    def test_synthetic_tree_prep(self):
        """
        Tests the synthetic tree preparation.
        """
        np.random.seed(6)

        # load the `Reactions` (built from 3 reaction templates)
        path_to_rxns = './data/ref/rxns_hb.json.gz'
        r_ref = ReactionSet()
        r_ref.load(path_to_rxns)
        rxns = r_ref.rxns

        # load the reference building blocks (1K here)
        path_to_building_blocks = './data/building_blocks_matched.csv.gz'
        building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()

        num_trials   = 14 
        num_finish   = 0
        num_error    = 0
        num_unfinish = 0

        trees = []
        for _ in tqdm(range(num_trials)):
            tree, action = synthetic_tree_generator(building_blocks, rxns, max_step=5)
            if action == 3:
                trees.append(tree)
                num_finish += 1
            elif action == -1:
                num_error += 1
            else:
                num_unfinish += 1

        synthetic_tree_set = SyntheticTreeSet(sts=trees)
        synthetic_tree_set.save('./data/st_data.json.gz')

        # check that the number of finished trees generated is == 10, and that
        # the number of unfinished trees generated is == 4
        self.assertEqual(num_finish, 10)
        self.assertEqual(num_unfinish, 4)
        
        # check here that the synthetic trees were correctly saved by
        # comparing to a provided reference file in 'SynNet/tests/data/ref/'
        sts_ref = SyntheticTreeSet()
        sts_ref.load('./data/ref/st_data.json.gz')
        for st_idx, st in enumerate(sts_ref.sts):
            st = st.__dict__
            ref_st = sts_ref.sts[st_idx].__dict__
            self.assertTrue(st == ref_st)

    def test_featurization(self):
        """
        Tests the featurization of the synthetic tree data into step-by-step
        data for training.
        """
        embedding='fp'
        radius=2
        nbits=4096
        dataset_type='train'

        path_st = './data/st_hb_test.json.gz'
        save_dir = './data/'
        reference_data_dir = './data/ref/'

        st_set = SyntheticTreeSet()
        st_set.load(path_st)
        data = st_set.sts
        del st_set

        states = []
        steps = []

        save_idx = 0
        for st in tqdm(data):
            try:
                state, step = organize(st, target_embedding=embedding, radius=radius, nBits=nbits)
            except Exception as e:
                print(e)
                continue
            states.append(state)
            steps.append(step)

        del data

        if len(steps) != 0:
            # save the states and steps
            states = sparse.vstack(states)
            steps = sparse.vstack(steps)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            sparse.save_npz(save_dir + 'states_' + str(save_idx) + '_' + dataset_type + '.npz', states)
            sparse.save_npz(save_dir + 'steps_' + str(save_idx) + '_' + dataset_type + '.npz', steps)

        # load the reference data, which we will compare against
        states_ref = sparse.load_npz(reference_data_dir + 'states_' + str(save_idx) + '_' + dataset_type + '.npz') 
        steps_ref = sparse.load_npz(reference_data_dir + 'steps_' + str(save_idx) + '_' + dataset_type + '.npz') 

        # check here that states and steps were correctly saved (need to convert the 
        # sparse arrays to non-sparse arrays for comparison)
        self.assertEqual(states.toarray().all(), states_ref.toarray().all())
        self.assertEqual(steps.toarray().all(), steps_ref.toarray().all())
    
    def test_dataprep(self):
        """
        Tests the training data preparation using the test subset data.
        """
        raise NotImplementedError
