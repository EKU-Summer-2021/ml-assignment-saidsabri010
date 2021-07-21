"""
this is the main model
"""
from src.svm import SupportVectorMachine

if __name__ == '__main__':
    instance = SupportVectorMachine({'C': [300, 500, 700],
                                     'gamma': [1, 0.1, 0.2],
                                     'kernel': ['rbf'],
                                     'epsilon': [0.2]
                                     })
    print(instance.plot_svm())
