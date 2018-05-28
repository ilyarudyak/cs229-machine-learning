import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, make_scorer

from charityML import split_data


def fit_model(X, y, params):
    dtc = DecisionTreeClassifier()
    k_folds = 10  # cs229 Lecture notes (2017) part VII
    scoring_fnc = make_scorer(fbeta_score, average='micro', beta=.5)

    grid = GridSearchCV(estimator=dtc, param_grid=params,
                        scoring=scoring_fnc, cv=k_folds)
    grid = grid.fit(X, y)

    return grid.best_estimator_


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(sample=.1)

    # max_depth_params = {'max_depth': np.arange(1, 11)}
    min_samples_split = {'min_samples_split': np.linspace(50, 600, 12).astype(int)}
    dtc_best = fit_model(X_train, y_train, min_samples_split)
    print(dtc_best.get_params()['min_samples_split'])
