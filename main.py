"""
this is the main model
"""
from src.svm import SupportVectorMachine

if __name__ == '__main__':
    instance = SupportVectorMachine()
    print(instance.run_grid_search())
