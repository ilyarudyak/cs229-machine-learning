import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer
from sklearn.model_selection import learning_curve
from time import time

from charityML import split_data


def fit_random_forest():
    rfc = RandomForestClassifier(n_estimators=500, max_depth=6, max_features=.7)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return accuracy_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=.5)


def plot_validation_curve():
    train_mean, test_mean = train_model()
    plt.plot(param_range, train_mean, 'o-', color='r', label='train')
    plt.plot(param_range, test_mean, 'o-', color='g', label='test')
    format_plot()


def format_plot():
    plt.xlabel(param_name)
    plt.ylabel('mean score: beta.5')
    plt.legend()
    plt.show()


def train_model():
    rfc = RandomForestClassifier(n_estimators=100, max_depth=6)
    train_scores, test_scores = validation_curve(
        rfc, X_train, y_train,
        cv=10,
        param_name=param_name, param_range=param_range,
        scoring=get_fbeta05_scoring())

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return train_mean, test_mean


def get_fbeta05_scoring():
    return make_scorer(fbeta_score, beta=.5)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(sample=1)

    start = time()
    accuracy, fbeta05 = fit_random_forest()
    print(f'random forest accuracy = {accuracy*100:.4f}% '
          f'beta .5 = {fbeta05*100:.4f}% '
          f'time={time()-start:.4f}s')

    # param_name = 'n_estimators'
    # param_range = np.linspace(100, 1000, 3).astype(int)

    # param_name = 'max_depth'
    # param_range = np.arange(1, 11)

    # param_name = 'max_features'
    # param_range = np.linspace(.1, 1, 10)
    #
    # plot_validation_curve()
