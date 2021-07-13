import numpy as np
from src import Polynomial

if __name__ == '__main__':
    coeffs = np.array([1,0,0])
    polynom = Polynomial(coeffs)
    print(polynom.evaluate(3))
    print(polynom.roots())