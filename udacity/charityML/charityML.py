import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

import visuals as vs

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def get_data():
    data = pd.read_csv("census.csv")
    features_raw = data.drop('income', axis=1)
    income = data.income.copy()
    income[data.income == '<=50K'] = 0
    income[data.income == '>50K'] = 1
    return data, features_raw, income


def data_exploration():
    n_records = income.shape[0]
    n_greater_50k = np.sum(income)
    n_at_most_50k = n_records - n_greater_50k
    greater_percent = n_greater_50k / n_records
    print(f'n_records={n_records:,} n_greater_50k={n_greater_50k:,} '
          f'n_at_most_50k={n_at_most_50k:,} greater_percent={greater_percent*100:.1f}%')


def data_preprocess():
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
    return features_final


if __name__ == '__main__':
    data, features_raw, income = get_data()
    features_final = data_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                        income,
                                                        test_size=0.2,
                                                        random_state=0)

    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

