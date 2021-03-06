import numpy as np

from plotting_data import *


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient(X, y, theta):
    m, n = X.shape
    z = (X.dot(theta)) * y
    grad = np.ones((n, 1))
    for j in range(n):
        s = 0
        for i in range(m):
            s += (sigmoid(z[i, :]) - 1) * y[i, :] * X[i, j]
        grad[j, :] = s
    return grad / m


def gradient_sum(X, y, theta):
    m, n = X.shape
    z = (X.dot(theta)) * y
    # grad = np.ones((n, 1))
    # for j in range(n):
    #     grad[j, :] = np.sum((sigmoid(z) - 1) * y * X[:, j:(j+1)])

    # this is broadcasting of (m, 1) to (m, n)
    grad = np.sum(((sigmoid(z) - 1) * y) * X, axis=0).reshape(n, 1)
    return grad / m


def gradient_vect(X, y, theta):
    m, _ = X.shape
    z = (X.dot(theta)) * y
    return X.T.dot((sigmoid(z) - 1) * y) / m


def hessian_sum(X, y, theta):
    m, n = X.shape
    z = (X.dot(theta)) * y
    print(z.shape)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = np.sum(sigmoid(z) * (1 - sigmoid(z))
                                        * X[:, i].reshape(m, 1)
                                        * X[:, j].reshape(m, 1))
    return H / m


def hessian_vect(X, y, theta):
    m, _ = X.shape
    z = (X.dot(theta)) * y
    d = (sigmoid(z) * (1 - sigmoid(z))).ravel()
    D = np.diag(d)
    return X.T.dot(D).dot(X) / m


def logistic_newton(X, y):
    theta = np.zeros((X.shape[1], 1))

    i = 0
    while i < 10:
        grad = gradient_vect(X, y, theta)
        hess = hessian_vect(X, y, theta)
        theta -= np.linalg.inv(hess).dot(grad)
        i += 1

    return theta


if __name__ == '__main__':
    X, y = get_data()
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    theta = logistic_newton(X, y)

    plot_data(X[:, 1:3], y)
    plot_decision_boundary(X[:, 1:3], theta)
    plt.show()
