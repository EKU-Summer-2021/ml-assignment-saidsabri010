"""
this is a unittest model
"""

import os
import unittest
import pandas as pd
import numpy as np
from src.mlp_classifier import MlpClassifier


class MyTestCase(unittest.TestCase):
    """
    test class
    """

    def setUp(self):
        data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
        self.data = MlpClassifier(
            data[['Pregnancies', 'Glucose', 'BloodPressure',
                                        'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
                                  data['Outcome'],
                                  {
                                      'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                                      'activation': ['tanh', 'relu', 'logistic'],
                                      'solver': ['lbfgs', 'sgd', 'adam'],
                                      'alpha': [0.0001, 0.05],
                                      'learning_rate': ['constant', 'adaptive'],
                                  }
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
        data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
        actual = self.data.plot_mlp(data[['Pregnancies', 'Glucose', 'BloodPressure',
                                        'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values.reshape(-1, 1),
                                    data['Outcome'].values.reshape(-1, 1))
        expected = 0.7
        boolean = np.isclose(actual, expected, rtol=0.3)
        self.assertEqual(boolean, True)
