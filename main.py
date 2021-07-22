"""
this is the main module
"""
import pandas as pd
from src.decisiontree import DecisionTree

dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
instance = DecisionTree({'criterion': ['gini', 'entropy'],
                          'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15,
                                        20, 30, 40, 50, 70, 90, 120, 150]},
                        dataframe[['Pregnancies', 'Glucose', 'BloodPressure',
                                    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
                        , dataframe['Outcome'].values.reshape(-1, 1))

print(instance.run_grid_search())
