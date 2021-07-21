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
        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        self.data = SupportVectorMachine({'C': [300, 500, 700],
                                          'gamma': [1, 0.1, 0.2],
                                          'kernel': ['rbf'],
                                          'epsilon': [0.2]
                                          }, dataframe['coarseaggregate'].values.reshape(-1, 1),
                                         dataframe['csMPa'].values.reshape(-1, 1))

    def test_run_grid_search(self):
        """
        test method : we test if the output is an instance of dataframe
        """
        result = self.data.run_grid_search()
        data = pd.DataFrame(result)
        self.assertIsInstance(data, pd.DataFrame)

    def test_save(self):
        """
               test method : we test if the output is an instance of dataframe
        """
        dir_path = os.getcwd()
        self.data.save()
        self.assertTrue(os.path.exists(dir_path))

    def test_plotting(self):
        """
        test method: we test if the score is close to the expected one
        """
        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        actual = self.data.plot_svm(dataframe['coarseaggregate'].values.reshape(-1, 1),
                                    dataframe['csMPa'].values.reshape(-1, 1))
        expected = 0.4
        boolean = np.isclose(actual, expected, rtol=0.3)
        self.assertEqual(boolean, True)
