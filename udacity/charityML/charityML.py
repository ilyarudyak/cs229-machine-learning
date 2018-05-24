import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

import visuals as vs


def get_data():
    data = pd.read_csv("census.csv")
    income = data.income.copy()
    income[data.income == '<=50K'] = 0
    income[data.income == '>50K'] = 1
    return data, income


def data_exploration():
    n_records = income.shape[0]
    n_greater_50k = np.sum(income)
    n_at_most_50k = n_records - n_greater_50k
    greater_percent = n_greater_50k / n_records
    print(f'n_records={n_records:,} n_greater_50k={n_greater_50k:,} '
          f'n_at_most_50k={n_at_most_50k:,} greater_percent={greater_percent*100:.1f}%')


if __name__ == '__main__':
    data, income = get_data()
    # data_exploration()

    vs.distribution(data)
    plt.show()
