import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from sklearn.metrics import accuracy_score


def get_data():
    X, y = make_blobs(n_samples=300, n_features=2,
                      centers=4, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def plot_data():
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
    plt.show()


def dt_classifier(max_depth=None):
    dtc = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
    plot_decision_regions(X_train, y_train,
                          clf=dtc,
                          legend=2,
                          markers='oooo^v')
    plt.show()


def plot_validation_curve():
    train_mean, test_mean = train_model()
    plt.plot(max_depth, train_mean, 'o-', color='r', label='train')
    plt.plot(max_depth, test_mean, 'o-', color='g', label='test')
    format_plot()


def format_plot():
    plt.xlabel('max_depth')
    plt.ylabel('mean score: accuracy')
    plt.legend()
    plt.show()


def train_model():
    model = DecisionTreeClassifier()
    train_scores, test_scores = validation_curve(
        model, X_train, y_train,
        cv=10,
        param_name='max_depth', param_range=max_depth,
        scoring='accuracy')

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return train_mean, test_mean


def fit_model():
    dtc = DecisionTreeClassifier()
    params = {'max_depth': range(1, 11)}
    grid = GridSearchCV(estimator=dtc, param_grid=params,
                        scoring='accuracy', cv=10)
    grid = grid.fit(X_train, y_train)
    return grid.best_estimator_


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    max_depth = np.arange(1, 11)

    # dt_classifier()
    # plot_validation_curve()
    # dt_optimized = fit_model()
    # print(f'optimized max_depth = {dt_optimized.get_params()["max_depth"]} '
    #       f'optimized accuracy = {accuracy_score(y_test, dt_optimized.predict(X_test))*100:.1f}%')
    dt_classifier(max_depth=4)
