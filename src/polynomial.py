'''
Polynomial module with a simple Polynomial example class.
'''
import numpy as np

class Polynomial():
    '''
    Polynomial object which can be evaluated.
    '''

    __coeffs = [0] # type: np.ndarray

    def __init__(self, coeffs : np.ndarray):
        self.__coeffs = coeffs

    def evaluate(self, x_value :float):
        '''
        Evaluate the Polynomial funcation ad the X position
        '''
        return np.polyval(self.__coeffs, x_value)

    def roots(self):
        '''
        Calculates the roots of the Polynomial function.
        :return:
        '''
        return np.roots(self.__coeffs)
