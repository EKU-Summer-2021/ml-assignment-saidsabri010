"""
this is a unittest model
"""
import unittest
import numpy as np
from src.svm import SupportVectorMachine


class MyTestCase(unittest.TestCase):
    """
    test class
    """

    def test_something(self):
        """
        test method
        """
        result = SupportVectorMachine()
        get_result = result.support()
        expected = 0.2
        if np.isclose(get_result, expected, rtol=0.2):
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
