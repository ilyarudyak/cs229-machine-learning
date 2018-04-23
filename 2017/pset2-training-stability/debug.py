from lr_debug import *


def different_lr():
    learning_rates = [.0001, .001, .01, .1]
    Xb, Yb = load_data('data_b.txt')
    for lr in learning_rates:
        logistic_regression(Xb, Yb, learning_rate=lr)


def scaling_lr():
    Xb, Yb = load_data('data_b.txt')
    logistic_regression(Xb, Yb, scaling=True)


if __name__ == '__main__':
    scaling_lr()
