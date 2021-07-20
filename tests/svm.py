"""
this is a unittest model
"""
import unittest
import os
import numpy as np
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

    def test_run_grid_search_output(self):
        """
            test method : we test if the output is close to the expected one
        """
        result = self.data.run_grid_search()
        data = pd.read_csv(r"C:\Users\HP\PycharmProjects\ml-assignment-saidsabri010\tests\StoredResults\score.csv")
        expected = pd.DataFrame(data)
        self.assertEqual(np.isclose(result, expected, rtol=0.7, atol=0), True)

    def test_save(self):
        """
               test method : we test if the output is an instance of dataframe
        """
        dir_path = os.getcwd()
        self.data.save()
        self.assertTrue(os.path.exists(dir_path))

    def test_plotting(self):
        """
        test method: we test if the score is higher than 0.2
        """
        actual = self.data.plot_svm()
        expected = 0.2
        self.assertGreater(actual, expected)
