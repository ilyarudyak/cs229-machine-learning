import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_digits


def get_digits_data():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=.25, random_state=0)
    return X_train, X_test, y_train, y_test


def fit_rf():
    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return y_pred


def plot_confusion_matrix():
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_digits_data()
    y_pred = fit_rf()

    # print(classification_report(y_test, y_pred))
    plot_confusion_matrix()

