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
    def test_something(self):
        """
        test method
        """
        file = SupportVectorMachine()
        data = pd.DataFrame(file.data2)
        self.assertIsInstance(data, pd.DataFrame)
