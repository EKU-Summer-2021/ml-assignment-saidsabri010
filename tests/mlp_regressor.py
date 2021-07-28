"""
this is a unittest model
"""

import os
import unittest
import pandas as pd
import numpy as np
from src.mlp_regressor import MLPRepressor


class MyTestCase(unittest.TestCase):
    """
       test class
    """

    def setUp(self):
        data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        self.data = MLPRepressor(
            data[['cement', 'slag', 'flyash',
                  'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']],
            data['csMPa'],
            {"hidden_layer_sizes": [(1,), (50,)], "activation": ["identity", "logistic", "tanh", "relu"],
             "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005, 0.0005]}
        )

    def test_run_grid_search(self):
        """
        test method : we test if the output is an instance of dataframe
        """
        result = self.data.run_mlp()
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
        data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        actual = self.data.plot_mlp(data[['cement', 'slag', 'flyash',
                                          'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']]
                                    .values.reshape(-1, 1))
        expected = 0.7
        boolean = np.isclose(actual, expected, rtol=0.3)
        self.assertEqual(boolean, True)
