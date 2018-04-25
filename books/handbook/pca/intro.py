import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

sns.set()


def get_data():
    np.random.seed(42)
    X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T
    plt.scatter(X[:, 0], X[:, 1])
    return X


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def draw_principal_components(X, k=2):
    pca = PCA(k)
    pca.fit(X)

    plt.plot(X[:, 0], X[:, 1], '.', alpha=0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal')
    plt.show()


def draw_projections(X, k=1):
    pca = PCA(k)
    pca.fit(X)

    X_pca = pca.transform(X)
    print("original shape: ", X.shape)
    print("transformed shape:", X_pca.shape)

    X_new = pca.inverse_transform(X_pca)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    X = get_data()
    draw_projections(X)
