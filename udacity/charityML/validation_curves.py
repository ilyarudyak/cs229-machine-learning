import numpy as np
import matplotlib.pyplot as plt

from charityML import split_data
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier


def plot_validation_curve(sample=.1,
                          param_name='max_depth',
                          scoring='accuracy',
                          max_depth=np.arange(1, 11)):
    train_mean, test_mean = train_model(sample, param_name, scoring, max_depth)
    plt.plot(max_depth, train_mean, 'o-', color='r', label='train')
    plt.plot(max_depth, test_mean, 'o-', color='g', label='test')
    format_plot(sample, param_name, scoring)
    plt.show()


def train_model(sample, param_name, scoring, max_depth):
    model = DecisionTreeClassifier()
    sample = int(sample * X_train.shape[0])
    train_scores, test_scores = validation_curve(
        model, X_train[:sample], y_train[:sample],
        cv=10,
        param_name=param_name, param_range=max_depth,
        scoring=scoring)

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return train_mean, test_mean


def format_plot(sample, param_name, scoring):
    plt.xlabel(param_name)
    plt.ylabel('mean score')
    plt.legend()
    plt.title(f'validation curve: scoring=\'{scoring}\'; sample={sample*100:.0f}%')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data()

    plot_validation_curve()
