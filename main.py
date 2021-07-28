"""
this is the main module
"""
import pandas as pd
from src.mlp_regressor import MLPRepressor

data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
data = pd.DataFrame(data)
instance = MLPRepressor(data[['cement', 'slag', 'flyash',
                              'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']]
                        ,
                        data['csMPa'],
                        {"hidden_layer_sizes": [(1,), (50,)], "activation": ["identity", "logistic", "tanh", "relu"],
                         "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005, 0.0005]}
                        )

print(instance.plot_mlp(data[['cement', 'slag', 'flyash',
                              'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']]
                        .values.reshape(-1, 1)))
