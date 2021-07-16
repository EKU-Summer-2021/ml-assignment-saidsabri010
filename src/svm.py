"""
   generate the model
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

param_grid = {'C': [0.1, 1, 10, 100, 300, 500, 700, 1000],
              'gamma': [1, 0.1, 0.2, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'],
              'epsilon': [0.2, 0.1]
              }


class SupportVectorMachine:
    """
    class for svm
    """

    def __init__(self):
        self.grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)

    def __str__(self):
        return self.__class__.__name__

    def run_grid_search(self):
        """
        this function read the data from csv file and train_test_split it
        """
        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        data_x = dataframe['coarseaggregate'].values.reshape(-1, 1)
        data_y = dataframe['csMPa'].values.reshape(-1, 1)
        train_x, test_x, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
        # this is for scaling
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        sc_x.fit_transform(data_x)
        sc_y.fit_transform(data_y)
        self.grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
        self.grid.fit(train_x, y_train)
        self.grid.score(test_x, y_test)
        # grid.best_estimator_
        return self.grid.cv_results_

    def save(self):
        """
          method for svr
        """
        if self.grid is None:
            get_score = self.run_grid_search()
        get_score = self.run_grid_search()
        # save the score in a csv file
        filename = "score.csv"
        # create directory structure
        directory = "StoredResults"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        complete_name = os.path.join(path, filename)
        is_file = os.path.isfile(complete_name)
        if is_file:
            pass
        else:
            os.mkdir(path)
        # create file inside the created directory structure
        get_score = pd.DataFrame(get_score)
        get_score.to_csv(complete_name)

        return get_score
