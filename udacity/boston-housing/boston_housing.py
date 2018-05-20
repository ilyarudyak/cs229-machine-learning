import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor

import visuals as vs

import locale
locale.setlocale(locale.LC_ALL, '')


def get_data():
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)
    return data, prices, features


def split_data(X, y):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)
    return X_train, X_test, y_train, y_test


def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(2, 5)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(r2_score)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values
    # 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(estimator=regressor, param_grid=params,
                        scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def get_prediction():
    return model.predict(client_data)


# https://stackoverflow.com/a/320951/2047442
def print_prediction():
    for i, p in enumerate(prediction):
        print(f'client{i+1}: price={locale.currency(p, grouping=True)}')


if __name__ == '__main__':
    data, prices, features = get_data()
    X_train, X_test, y_train, y_test = split_data(features, prices)

    # vs.ModelLearning(X_train, y_train)
    vs.ModelComplexity(X_train, y_train)
    plt.show()

    # model = fit_model(X_train, y_train)
    # client_data = [[5, 17, 15],   # Client 1
    #                 [4, 32, 22],  # Client 2
    #                 [8, 3, 12]]   # Client 3
    # prediction = get_prediction()
    # print_prediction()

    # vs.PredictTrials(features, prices, fit_model, client_data)

