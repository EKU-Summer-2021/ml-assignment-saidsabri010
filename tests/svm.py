"""
this is a unittest model
"""
import unittest
import pandas as pd
from src.svm import SupportVectorMachine


class MyTestCase(unittest.TestCase):
    """
    test class
    """

    def setUp(self):
        self.data = SupportVectorMachine()

    def test_run_grid_search(self):
        """
        test method : we test if the output is an instance of dataframe
        """
        result = self.data.run_grid_search()
        data = pd.DataFrame(result)
        self.assertIsInstance(data, pd.DataFrame)


    #def test_run_grid_search_output(self):
        #result = self.data.run_grid_search()
        #expected = self.data.grid.cv_results_
        #np.testing.assert_allclose(result, expected, rtol=0.2, atol=0)

    #def test_save(self):
        #actual_get_score = self.data.save()
        #expected_get_score = pd.DataFrame
        #self.assertIsInstance(actual_get_score, expected_get_score)
