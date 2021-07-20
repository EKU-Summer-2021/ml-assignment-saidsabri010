"""
   generate the model
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# here i reduced the number of parameters because it takes too much time to get the results
param_grid = {'C': [300, 500, 700],
              'gamma': [1, 0.1, 0.2],
              'kernel': ['rbf'],
              'epsilon': [0.2]
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
        scalar = StandardScaler()
        data_x = preprocessing.scale(data_x)
        data_y = preprocessing.scale(data_y)
        scalar.fit_transform(data_x)
        scalar.fit_transform(data_y)
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

    def plot_svm(self):
        """
        this function plot the results and save it and return the score
        """
        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        data_x = dataframe['coarseaggregate'].values.reshape(-1, 1)
        data_y = dataframe['csMPa'].values.reshape(-1, 1)
        train_x, test_x, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
        self.grid.fit(train_x, y_train)
        plt.scatter(test_x, y_test)
        plt.plot(train_x, self.grid.predict(y_train), color="blue")
        plt.title('Support vector machine')

        # save plot
        filename = "score.csv"
        # create directory structure
        directory = "StoredResults"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        complete_name = os.path.join(path, filename)
        plt.savefig(complete_name + 'plot.png')
        plt.show()
        return self.grid.score(train_x, y_train)
