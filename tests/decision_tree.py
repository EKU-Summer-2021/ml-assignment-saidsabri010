"""
this is a unittest model
"""
import unittest
import os
import pandas as pd
from src.decisiontree import DecisionTree


class MyTestCase(unittest.TestCase):
    """
    test class
    """

    def setUp(self):
        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
        self.data = DecisionTree({'criterion': ['gini', 'entropy'],
                                   'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15,
                                                 20, 30, 40, 50, 70, 90, 120, 150]},
                                 dataframe[['Pregnancies', 'Glucose', 'BloodPressure',
                                             'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
                                 dataframe['Outcome'].values.reshape(-1, 1))

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
        actual = self.data.plot_decision()
        expected = 0.6
        self.assertGreater(actual, expected)
