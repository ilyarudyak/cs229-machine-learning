import numpy as np
import matplotlib.pyplot as plt


def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X


def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y


def normalize(X_):
    a, b = X_[:, 0], X_[:, 1]
    X = np.zeros_like(X_)
    X[:, 0] = (a - np.mean(a)) / np.std(a)
    X[:, 1] = (b - np.mean(b)) / np.std(b)
    return X


def load_data_normalized(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y


def calc_grad(X, Y, theta):
    m, n = X.shape
    # grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1. / m) * (X.T.dot(probs * Y))

    return grad


def calc_grad2(X, Y, theta):
    m, n = X.shape
    # grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(-margins))
    grad = -(1. / m) * (X.T.dot((1 - probs) * Y))

    return grad


def get_loss(X, Y, theta):
    margins = Y * X.dot(theta)
    log_probs = np.log(1. / (1 + np.exp(-margins)))

    return -np.mean(log_probs)


def logistic_regression(X, Y,
                        learning_rate=10,
                        max_iterations=1e5,
                        scaling=False,
                        verbose=False):
    m, n = X.shape
    theta = np.zeros(n)

    size = int(max_iterations / 10000)
    loss = np.zeros(size)

    i = 0
    while True:
        i += 1
        prev_theta = theta
        # grad = calc_grad(X, Y, theta)
        grad = calc_grad2(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            loss[int(i / 10000) - 1] = get_loss(X, Y, theta)
            if scaling:
                learning_rate /= i ^ 2
            if verbose:
                print('Finished iterations:', i, 'loss:', loss)
        if i > max_iterations:
            print('lr =', learning_rate, np.linalg.norm(prev_theta - theta))
            break
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return loss


def main():
    # print('==== Training model on data set A ====')
    # Xa, Ya = load_data('data_a.txt')
    # logistic_regression(Xa, Ya, verbose=True)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('data_b.txt')
    loss = logistic_regression(Xb, Yb, max_iterations=1e5)

    plt.plot(loss)
    plt.show()

    return


if __name__ == '__main__':
    main()
