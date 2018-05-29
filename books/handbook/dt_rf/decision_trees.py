import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions


def get_data():
    X, y = make_blobs(n_samples=300, n_features=2,
                      centers=4, random_state=0)
    return X, y


def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    plt.show()


def dt_classifier():
    dtc = DecisionTreeClassifier().fit(X, y)
    plot_decision_regions(X, y,
                          clf=dtc,
                          legend=2,
                          markers='oooo^v')
    plt.show()


if __name__ == '__main__':
    X, y = get_data()
    dt_classifier()
