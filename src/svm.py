"""
   generate the model
"""
import os
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

    def support(self):
        """
          function for svr
          """

        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        X = dataframe['coarseaggregate'].values.reshape(-1, 1)
        y = dataframe['csMPa'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # this is for scaling
        scaler = StandardScaler()
        sc_X = scaler
        sc_y = scaler
        sc_X.fit_transform(X)
        sc_y.fit_transform(y)
        param_grid = {'C': [0.1, 1, 10, 100, 300, 500, 700, 1000],
                      'gamma': [1, 0.1, 0.2, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf'],
                      'epsilon': [0.2, 0.1]
                      }
        grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)
        score = grid.score(X_test, y_test)
        # save the score in a csv file
        filename = "score.csv"
        # create directory structure
        directory = "StoredResults"
        parent_dir = "C:/Users/HP/PycharmProjects/ml-assignment-saidsabri010/"
        path = os.path.join(parent_dir, directory)
        completeName = os.path.join(path, filename)
        isFile = os.path.isfile(completeName)
        if isFile:
            pass
        else:
            os.mkdir(path)
        # create file inside the created directory structure
        file1 = open(completeName, "a")
        file1.write("\n")
        file1.write(str(score))

        return score
