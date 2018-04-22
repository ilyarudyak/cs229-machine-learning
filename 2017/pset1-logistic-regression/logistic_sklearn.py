from sklearn.linear_model import LogisticRegression
from plotting_data import *


def logistic_sklearn(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    theta = np.append(lr.intercept_, lr.coef_)
    return theta


if __name__ == '__main__':
    X, y = get_data()
    theta = logistic_sklearn(X, y)

    plot_data(X, y)
    plot_decision_boundary(X, y, theta)
    plt.show()
