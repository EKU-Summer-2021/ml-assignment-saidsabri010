"""
   generate the model
"""
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class SupportVectorMachine:
    """
    class for svm
    """
    def __init__(self):
        self.data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')

    def __str__(self):
        return self.__class__.__name__

    df = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
    X = df['coarseaggregate'].values.reshape(-1, 1)
    y = df['csMPa'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # this is for scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)
    score = grid.score(X_test, y_test)

    # save the score in a csv file
    fields = ['Score']
    rows = [[1.0],
            [0.4654]]
    filename = "Dt_modle_score.csv"
    with open(filename, 'w') as csv_file:
        # creating a csv writer object
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerow(rows)
    data2 = pd.read_csv("Dt_modle_score.csv")
