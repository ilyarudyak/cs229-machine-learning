import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


def get_data():
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.60, random_state=0)
    return X, y_true


def show_data(X):
    plt.scatter(X[:, 0], X[:, 1], s=20)
    plt.show()


def get_labels_kmeans(X, k=4):
    kmeans = KMeans(k).fit(X)
    return kmeans.predict(X)


def show_data_kmeans(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='tab10')
    plt.show()


def find_clusters(X, k=4, rseed=2):
    # 1. Randomly choose clusters
    np.random.seed(rseed)
    index = np.random.permutation(X.shape[0])[:k]
    centers = X[index, :]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = [np.mean(X[labels == i, :], axis=0) for i in range(k)]

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

        return centers, labels


if __name__ == '__main__':
    X, _ = get_data()

    for i in range(10):
        _, y = find_clusters(X, rseed=i)
        show_data_kmeans(X, y)


