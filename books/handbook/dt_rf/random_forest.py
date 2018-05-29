import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

from dt_rf.decision_trees import get_data

sns.set()

from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from sklearn.metrics import accuracy_score


def plot_decision_boundaries(n_estimators=100):
    rfc = RandomForestClassifier(n_estimators=n_estimators,
                                 random_state=0).fit(X_train, y_train)
    plot_decision_regions(X_train, y_train,
                          clf=rfc,
                          legend=2,
                          markers='oooo^v')
    print(f'accuracy n_estimators=100: {accuracy_score(y_test, rfc.predict(X_test))*100:.1f}%')
    plt.show()


def plot_validation_curve():
    train_mean, test_mean = train_model()
    plt.plot(n_estimators, train_mean, 'o-', color='r', label='train')
    plt.plot(n_estimators, test_mean, 'o-', color='g', label='test')
    format_plot()


def format_plot():
    plt.xlabel('n_estimators')
    plt.ylabel('mean score: accuracy')
    plt.legend()
    plt.show()


def train_model():
    model = RandomForestClassifier()
    train_scores, test_scores = validation_curve(
        model, X_train, y_train,
        cv=10,
        param_name='n_estimators', param_range=n_estimators,
        scoring='accuracy')

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    return train_mean, test_mean


def optimize_model():
    dtc = RandomForestClassifier(random_state=0)
    params = {'n_estimators': n_estimators}
    grid = GridSearchCV(estimator=dtc, param_grid=params,
                        scoring='accuracy', cv=10)
    grid = grid.fit(X_train, y_train)
    return grid.best_estimator_


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    # plot_decision_boundaries()
    n_estimators = np.array([1, 10, 50, 100, 200])
    plot_validation_curve()
