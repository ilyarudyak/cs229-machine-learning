import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
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


def plot_one_curve(depth):
    sizes, train_mean, test_mean = train_model(depth)
    plt.plot(sizes, train_mean, 'o-', color='b', label='training score')
    plt.plot(sizes, test_mean, 'o-', color='r', label='testing score')
    plt.show()


def plot_multiple_curves(depths):
    _, ax = plt.subplots(1, depths.shape[0], figsize=(12.5, 2.5))
    for j, depth in enumerate(depths):
        sizes, train_mean, test_mean = train_model(depth)
        ax[j].plot(sizes, train_mean, 'o-', color='b', label='train')
        ax[j].plot(sizes, test_mean, 'o-', color='r', label='test')
        ax[j].set_title(f'depth={depth}')
    plt.legend(loc=(1.04, 0))
    plt.show()


def train_model(depth):
    model = DecisionTreeRegressor(max_depth=depth)
    sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=10,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2')

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return sizes, train_mean, test_mean


if __name__ == '__main__':
    data, prices, features = get_data()
    X_train, X_test, y_train, y_test = split_data(features, prices)

    depths = np.array([1, 3, 4, 10])
    plot_multiple_curves(depths)
