"""
this is the main module
"""
import pandas as pd
import numpy as np
from src.decisiontree import DecisionTree

dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
instance = DecisionTree({'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15)},
                        dataframe[['Pregnancies', 'Glucose', 'BloodPressure',
                                   'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
                        ,
                        dataframe['Outcome'])

print(instance.plot_decision())
