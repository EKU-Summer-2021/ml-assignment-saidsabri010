"""
this is the main module
"""
import pandas as pd
from src.mlp_classifier import MlpClassifier

data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
data = pd.DataFrame(data)
instance = MlpClassifier(data[['Pregnancies', 'Glucose', 'BloodPressure',
                               'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
                         data['Outcome'],
                         {
                             'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                             'activation': ['tanh', 'relu', 'logistic'],
                             'solver': ['lbfgs', 'sgd', 'adam'],
                             'alpha': [0.0001, 0.05],
                             'learning_rate': ['constant', 'adaptive', 'invscaling'],
                             'learning_rate_init': [0.001, 0.002, 0.003]
                         }
                         )

print(instance.plot_mlp(data[['Pregnancies', 'Glucose', 'BloodPressure',
                              'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
                        .values.reshape(-1, 1),
                        data['Outcome'].values.reshape(-1, 1)))
