import numpy as np
import pandas as pd

from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score, fbeta_score


def get_raw_data():
    data = pd.read_csv("census.csv")
    features_raw = data.drop('income', axis=1)
    income = data.income.copy()
    income[data.income == '<=50K'] = 0
    income[data.income == '>50K'] = 1
    income = income.astype('int')
    return data, features_raw, income


def explore_data():
    data, features_raw, income = get_raw_data()
    n_records = income.shape[0]
    n_greater_50k = np.sum(income)
    n_at_most_50k = n_records - n_greater_50k
    greater_percent = n_greater_50k / n_records
    print(f'n_records={n_records:,} n_greater_50k={n_greater_50k:,} '
          f'n_at_most_50k={n_at_most_50k:,} greater_percent={greater_percent*100:.1f}%')


def preprocess_data():
    data, features_raw, income = get_raw_data()
    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    # vs.distribution(features_log_transformed, transformed=True)

    # Normalize numerical features
    scaler = MinMaxScaler()  # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    # One-hot encode
    features_final = pd.get_dummies(features_log_minmax_transform)
    return income, features_final


def split_data(sample=.1):
    income, features_final = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features_final, income, test_size=0.2, random_state=0)
    sample = int(sample * X_train.shape[0])
    return X_train[:sample], X_test, y_train[:sample], y_test


def fit_decision_tree():
    dt = DecisionTreeClassifier(max_depth=6)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return accuracy_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=.5)


def fit_gaussian_nb():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    return accuracy_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=.5)


def fit_svm():
    svc = svm.SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    return accuracy_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=.5)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(sample=.1)
    print(X_train.shape)

    # start = time()
    # accuracy, fbeta05 = fit_decision_tree()
    # print(f'decision tree accuracy = {accuracy*100:.4f}% '
    #       f'beta .5 = {fbeta05*100:.4f}% '
    #       f'time={time()-start:.4f}s')

    # start = time()
    # accuracy, fbeta05 = fit_gaussian_nb()
    # print(f'gaussian NB accuracy = {accuracy*100:.1f}% '
    #       f'beta .5 = {fbeta05*100:.1f}% '
    #       f'time={time()-start:.4f}s')

    start = time()
    accuracy, fbeta05 = fit_svm()
    print(f'svm accuracy = {accuracy*100:.1f}% '
          f'beta .5 = {fbeta05*100:.1f}%'
          f'time={time()-start:.4f}s')

