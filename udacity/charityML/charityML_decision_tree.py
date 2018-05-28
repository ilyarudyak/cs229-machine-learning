import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import learning_curve

from charityML import split_data


def fit_model(X, y, params):
    dtc = DecisionTreeClassifier()
    k_folds = 10  # cs229 Lecture notes (2017) part VII
    scoring = get_fbeta05_scoring()

    grid = GridSearchCV(estimator=dtc, param_grid=params,
                        scoring=scoring, cv=k_folds)
    grid = grid.fit(X, y)

    return grid.best_estimator_


def plot_learning_curve():
    sizes, train_mean, test_mean = train_model()
    plt.plot(sizes, train_mean, 'o-', color='b', label='train')
    plt.plot(sizes, test_mean, 'o-', color='r', label='test')
    format_plot()
    plt.show()


def format_plot():
    plt.legend()
    plt.xlabel('size of training data')
    plt.ylabel('fbeta .5')
    plt.title('learning curve for DT(max_depth=6)')


def train_model():
    k_folds = 10  # cs229 Lecture notes (2017) part VII
    scoring = get_fbeta05_scoring()
    dtc = DecisionTreeClassifier(max_depth=6)
    sizes, train_scores, test_scores = learning_curve(
        dtc, X_train, y_train, cv=k_folds,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring)

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return sizes, train_mean, test_mean


def get_fbeta05_scoring():
    return make_scorer(fbeta_score, beta=.5)


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = split_data(sample=.1)
    #
    # max_depth_params = {'max_depth': np.arange(1, 11)}
    # dtc_best = fit_model(X_train, y_train, max_depth_params)
    # min_samples_split = {'min_samples_split': np.linspace(50, 600, 12).astype(int)}
    # dtc_best = fit_model(X_train, y_train, min_samples_split)
    # print(dtc_best.get_params()['min_samples_split'])

    X_train, X_test, y_train, y_test = split_data(sample=1)
    plot_learning_curve()
