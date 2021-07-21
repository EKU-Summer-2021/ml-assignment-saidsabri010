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


class SupportVectorMachine:
    """
    class for svm
    """

    def __init__(self, param_grid, data_x, data_y):
        self.grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
        self.train_x, self.test_x, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.2)

    def __str__(self):
        return self.__class__.__name__

    def run_grid_search(self):
        """
        this function read the data from csv file and train_test_split it
        """
        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')
        data_x = dataframe['coarseaggregate'].values.reshape(-1, 1)
        data_y = dataframe['csMPa'].values.reshape(-1, 1)
        # this is for scaling
        scalar = StandardScaler()
        data_x = preprocessing.scale(data_x)
        data_y = preprocessing.scale(data_y)
        scalar.fit_transform(data_x)
        scalar.fit_transform(data_y)
        self.grid.fit(self.train_x, self.y_train)
        self.grid.score(self.test_x, self.y_test)
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
        filename = "results.csv"
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

    def plot_svm(self, data_x, data_y):
        """
        this function plot the results and save it and return the score
        """
        scalar = StandardScaler()
        scalar.fit_transform(data_x)
        scalar.fit_transform(data_y)
        self.grid.fit(self.train_x, self.y_train)
        y_pred = self.grid.predict(self.test_x)
        scalar.inverse_transform(y_pred)
        plt.scatter(self.y_test, y_pred)
        plt.title('Support vector machine')
        # save plot
        filename = "results.csv"
        # create directory structure
        directory = "StoredResults"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        complete_name = os.path.join(path, filename)
        plt.savefig(complete_name + 'plot.png')
        plt.show()
        plt.scatter(self.test_x, self.y_test)
        plt.scatter(self.train_x, self.y_train, color="red")
        plt.savefig(complete_name + 'plot.png')
        plt.show()
        return self.grid.score(self.train_x, self.y_train)
