"""
this is the main model
"""
import pandas as pd
from src.svm import SupportVectorMachine

if __name__ == '__main__':
    dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
    instance = SupportVectorMachine({'C': [300, 500, 700],
                                     'gamma': [1, 0.1, 0.2],
                                     'kernel': ['rbf'],
                                     'epsilon': [0.2]
                                     }, dataframe['coarseaggregate'].values.reshape(-1, 1),
                                    dataframe['csMPa'].values.reshape(-1, 1))
    print(instance.plot_svm(dataframe['coarseaggregate'].values.reshape(-1, 1),
                            dataframe['csMPa'].values.reshape(-1, 1)))
