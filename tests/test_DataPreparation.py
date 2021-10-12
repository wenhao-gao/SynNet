"""
Unit tests for the data preparation.
"""
import unittest
from syn_net.utils.data_utils import SyntheticTreeSet
import os
from tqdm import tqdm
from scipy import sparse


class TestDataPrep(unittest.TestCase):
    """
    Tests for the data preparation: (1) data splitting, (2) featurization, (3) training
    data preparation for each network.
    """
    def test_datasplits(self):
        """
        Tests the data splitting. TODO not sure if this is worth testing
        """
        raise NotImplementedError

    def test_featurization(self):
        """
        Tests the featurization of the synthetic tree data into step-by-step
        data for training.
        """
        from scripts.st2steps import organize
        num_save=999999999999
        embedding='fp'
        radius=2
        nbits=4096
        dataset_type='train'

        path_st = f"./tests/data/st_hb_test.json.gz"
        save_dir = f"./tests/data/"
        reference_data_dir = f"./tests/data/ref/"

        st_set = SyntheticTreeSet()
        st_set.load(path_st)
        print('Original length: ', len(st_set.sts))
        data = st_set.sts
        del st_set
        print('Working length: ', len(data))

        states = []
        steps = []

        idx = 0
        save_idx = 0
        for st in tqdm(data):
            try:
                state, step = organize(st, target_embedding=embedding, radius=radius, nBits=nbits)
            except Exception as e:
                print(e)
                continue
            states.append(state)
            steps.append(step)
            idx += 1
            if idx % num_save == 0:
                print('Saving......')
                states = sparse.vstack(states)
                steps = sparse.vstack(steps)
                sparse.save_npz(save_dir + 'states_' + str(save_idx) + '_' + dataset_type + '.npz', states)
                sparse.save_npz(save_dir + 'steps_' + str(save_idx) + '_' + dataset_type + '.npz', steps)
                save_idx += 1
                del states
                del steps
                states = []
                steps = []

        del data
    
        if len(steps) != 0:
            states = sparse.vstack(states)
            steps = sparse.vstack(steps)

            print('Saving......')
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
