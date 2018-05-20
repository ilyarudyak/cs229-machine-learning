import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, validation_curve
from sklearn.tree import DecisionTreeRegressor


def get_data():
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)
    return data, prices, features


def split_data(X, y):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_one_curve():
    train_mean, test_mean = train_model()
    plt.plot(max_depth, train_mean, 'o-', color='r', label='train')
    plt.plot(max_depth, test_mean, 'o-', color='g', label='test')
    plt.legend()
    plt.show()


def train_model():
    model = DecisionTreeRegressor()
    train_scores, test_scores = validation_curve(
        model, X_train, y_train,
        cv=10,
        param_name='max_depth', param_range=max_depth,
        scoring='r2')

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return train_mean, test_mean


if __name__ == '__main__':
    data, prices, features = get_data()
    X_train, X_test, y_train, y_test = split_data(features, prices)

    max_depth = np.arange(1, 11)
    plot_one_curve()
