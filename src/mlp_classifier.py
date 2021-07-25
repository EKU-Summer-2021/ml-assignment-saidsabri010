"""
this module is for mlp classifier
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


class MlpClassifier:
    """
    this class is for the mlp classifier
    """

    def __init__(self, data_x, data_y, param_grid):
        self.data_x = data_x
        self.data_y = data_y
        self.grid = GridSearchCV(MLPClassifier(), param_grid,  n_jobs=-1, cv=3)
        self.train_x, self.test_x, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.2)

    def __str__(self):
        return self.__class__.__name__

    def run_mlp(self):
        """
        this function will run our neural network
        """
        self.grid.fit(self.train_x, self.y_train)
        y_pred = self.grid.predict(self.test_x)
        metrics.accuracy_score(self.y_test, y_pred)
        return self.grid.cv_results_

    def save(self):
        """
          method for saving the csv results
        """
        if self.grid is None:
            get_score = self.run_mlp()
        get_score = self.run_mlp()
        # save the score in a csv file
        filename = "mlp_result.csv"
        # create directory structure
        directory = "MLPStoredResults"
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

    def plot_mlp(self, data_x, data_y):
        """
        this function plot the results and save it and return the score
        """
        scalar = StandardScaler()
        scalar.fit_transform(data_x)
        scalar.fit_transform(data_y)
        self.grid.fit(self.train_x, self.y_train)
        plt.plot(self.train_x, self.y_train, color='green', alpha=0.8, label='Train')
        plt.plot(self.test_x, self.y_test, color='magenta', alpha=0.8, label='Test')
        plt.title('Multi layer Perceptron')
        plt.xlabel('Epochs')
        plt.legend(loc='upper left')

        # save plot
        filename = "MLPResults.csv"
        # create directory structure
        directory = "MLPStoredResults"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        complete_name = os.path.join(path, filename)
        plt.savefig(complete_name + 'mlpplot.png')
        plt.show()
        return self.grid.score(self.train_x, self.y_train)
