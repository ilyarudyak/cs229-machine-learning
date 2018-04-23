from lr_debug import *


def different_lr():
    learning_rates = [.0001, .001, .01, .1]
    Xb, Yb = load_data('data_b.txt')
    for lr in learning_rates:
        logistic_regression(Xb, Yb, learning_rate=lr)


def scaling_lr():
    Xb, Yb = load_data('data_b.txt')
    logistic_regression(Xb, Yb, scaling=True)


def normalizes_lr():
    Xb, Yb = load_data_normalized('data_b.txt')
    logistic_regression(Xb, Yb, learning_rate=1e-5, max_iterations=1e6)


if __name__ == '__main__':
    normalizes_lr()
