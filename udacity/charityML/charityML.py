import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

import visuals as vs

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score


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


def split_data():
    income, features_final = preprocess_data()
    return train_test_split(features_final, income, test_size=0.2, random_state=0)


def fit_decision_tree():
    dt = DecisionTreeClassifier(min_samples_split=50)
    dt.fit(X_train, y_train)
    return accuracy_score(y_test, dt.predict(X_test))


def fit_gaussian_nb():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return accuracy_score(y_test, gnb.predict(X_test))


def fit_svm():
    svc = svm.SVC()
    svc.fit(X_train, y_train)
    return accuracy_score(y_test, svc.predict(X_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data()
    print(X_train.shape)

    # print(f'decision tree accuracy = {fit_decision_tree()*100:.1f}%')
    # print(f'gaussian NB accuracy = {fit_gaussian_nb()*100:.1f}%')
    # print(f'svm accuracy = {fit_svm()*100:.1f}%')

