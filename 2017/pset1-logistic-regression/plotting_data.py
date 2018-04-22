import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def get_data():
    X = pd.read_csv('logistic_x.txt', '\s+', header=None).as_matrix()
    y = pd.read_csv('logistic_y.txt', header=None).as_matrix()
    return X, y


def plot_data(X, y):
    y = y.reshape(X.shape[0])
    X_pos, X_neg = X[y == 1, :], X[y == -1, :]

    plt.plot(X_pos[:, 0], X_pos[:, 1], 'r.')
    plt.plot(X_neg[:, 0], X_neg[:, 1], 'b.')


def plot_data_and_show(X, y):
    plot_data(X, y)
    plt.show()


def plot_decision_boundary(X,theta):
    t0, t1, t2 = theta
    x1 = np.linspace(np.floor(np.min(X[:, 0])), np.floor(np.max(X[:, 0])), 100)
    x2 = -(t0 + t1 * x1) / t2
    plt.plot(x1, x2, 'k')


if __name__ == '__main__':
    X, y = get_data()
    plot_data_and_show(X, y)