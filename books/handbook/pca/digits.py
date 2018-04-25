import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits


def get_digits(k=2):
    digits = load_digits()
    print(digits.data.shape)
    pca = PCA(2)
    projected = pca.fit_transform(digits.data)
    print(digits.data.shape)
    print(projected.shape)
    return digits, projected


def visualize_digits(digits, projected):
    plt.scatter(projected[:, 0], projected[:, 1],
                c=digits.target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()


def choose_k(digits):
    pca = PCA().fit(digits.data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


if __name__ == '__main__':
    sns.set()
    digits, projected = get_digits()
    choose_k(digits)



