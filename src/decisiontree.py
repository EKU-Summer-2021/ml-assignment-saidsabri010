"""
this module is for decision tree
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    """
    decision tree class
    """

    def __init__(self, param_grid, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit=True)
        self.train_x, self.test_x, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.2)

    def __str__(self):
        return self.__class__.__name__

    def run_grid_search(self):
        """
        this function will clean,read,split into train and test
        """
        dataframe = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/credit_card_dataset/main/diabetes.csv')
        data_x = dataframe[['Pregnancies', 'Glucose', 'BloodPressure',
                            'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        scalar = StandardScaler()
        data_x = preprocessing.scale(data_x)
        # data_y = preprocessing.scale(data_y)
        scalar.fit_transform(data_x)
        # data_y = scalar.fit_transform(data_y)
        self.grid.fit(self.train_x, self.y_train)
        self.grid.score(self.test_x, self.y_test)
        # y_pred = self.grid.predict(self.test_x)
        # return metrics.accuracy_score(y_test, y_pred)
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

    def plot_decision(self):
        """
        this function plot the results and save it and return the score
        """
        scalar = StandardScaler()
        scalar.fit_transform(self.data_x)
        scalar.fit_transform(self.data_y)
        self.grid.fit(self.train_x, self.y_train)
        y_pred = self.grid.predict(self.test_x)
        plt.scatter(self.y_test, y_pred)
        plt.title('Decision Tree')
        # save plot
        filename = "results.csv"
        # create directory structure
        directory = "StoredResults"
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        complete_name = os.path.join(path, filename)
        plt.savefig(complete_name + 'diabetes.png')
        plt.show()
        # plot tree
        clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(self.train_x, self.y_train)
        plt.figure(figsize=(15, 10))
        tree.plot_tree(clf,
                       max_depth=3,
                       rounded=True,
                       filled=True)

        plt.savefig(complete_name + 'plot.png')
        plt.show()
        return self.grid.score(self.train_x, self.y_train)
