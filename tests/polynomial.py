import unittest
import numpy as np
from src import Polynomial


class PolynomialTets(unittest.TestCase):

    def test_square_of_three_should_be_nince(self):
        # given
        p = Polynomial(np.array([1,0,0]))
        X_VALUE = 3
        EXPECTED = 9
        # when
        ACTUAL = p.evaluate(X_VALUE)
        # then
        self.assertEqual(EXPECTED, ACTUAL)

    # def test_fails(self):
    #     self.assertEqual(True, False)