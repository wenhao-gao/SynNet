"""
Unit tests for the model predictions.
"""
import unittest


class TestPredict(unittest.TestCase):
    """
    Tests for model predictions: (1) greedy search, (2) beam search for 
    reactant 1 only, (3) beam search for the full tree.
    """
    def test_predict(self):
        """
        Tests synthetic tree generation given a molecular embedding. No beam search.
        """
        raise NotImplementedError

    def test_predict_beam_rt1(self):
        """
        Tests synthetic tree generation given a molecular embedding. Uses beam
        search for reactant 1.
        """
        raise NotImplementedError

    def test_predict_beam_full_tree(self):
        """
        Tests synthetic tree generation given a molecular embedding. Uses beam
        search for generating the full tree.
        """
        raise NotImplementedError
