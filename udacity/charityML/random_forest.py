import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import learning_curve
from time import time

from charityML import split_data


def fit_random_forest():
    rfc = RandomForestClassifier(n_estimators=100, max_depth=6)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return accuracy_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=.5)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(sample=1)

    start = time()
    accuracy, fbeta05 = fit_random_forest()
    print(f'random forest accuracy = {accuracy*100:.4f}% '
          f'beta .5 = {fbeta05*100:.4f}% '
          f'time={time()-start:.4f}s')
