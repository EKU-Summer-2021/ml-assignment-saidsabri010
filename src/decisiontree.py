"""
this module is for decision tree
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:  # pylint: disable= R0902
    """
    decision tree class
    """

    def __init__(self, param_grid, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.grid = None
        self.param_grid = param_grid
        self.train_x, self.test_x, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.2)

    def __str__(self):
        return self.__class__.__name__

    def run_grid_search(self):
        """
        this function will clean,read,split into train and test
        """
        self.grid = GridSearchCV(DecisionTreeClassifier(), self.param_grid, refit=True)
        self.grid.fit(self.train_x, self.y_train)
        self.grid.score(self.test_x, self.y_test)
        return self.grid.cv_results_

    def save(self):
        """
          this function saves the results
        """
        if self.grid is None:
            get_score = self.run_grid_search()
        get_score = self.run_grid_search()
        # save the score in a csv file
        filename = "results.csv"
        # create directory structure
        directory = "StoredDecisionTreeResults"
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

    def plot_decision(self):
        """
        this function plot the results and save it and return the score
        """
        if self.grid is None:
            self.run_grid_search()
        plot_confusion_matrix(self.grid.best_estimator_, self.test_x, self.y_test)
        # save plot
        filename = "results.csv"
        # create directory structure
        directory = "StoredDecisionTreeResults"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        complete_name = os.path.join(path, filename)
        plt.savefig(complete_name + 'plot.pdf')
        plt.show()
        tree.plot_tree(self.grid.best_estimator_, feature_names=self.train_x.columns, class_names=['no', 'yes'],
                       filled=True)
        plt.savefig(complete_name + 'plot.png')
        plt.show()
        return self.grid.score(self.train_x, self.y_train)
