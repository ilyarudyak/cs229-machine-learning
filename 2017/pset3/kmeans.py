import pandas as pd
import numpy as np

from draw_clusters import draw_clusters


def closest_centroid(centroids, X):
    m, _ = X.shape
    return np.array([np.argmin(np.linalg.norm(centroids - X[i, :], axis=1))
                     for i in range(m)])


def recompute_centroids(X, k, clusters_index):
    return np.array([np.mean(X[clusters_index == i, :], axis=0)
                     for i in range(k)])


def k_means(X, k=3):
    m, n = X.shape
    old_centroids = np.zeros((k, n))
    index = np.random.permutation(np.arange(m))[:k]
    centroids = X[index, :]
    clusters_index = np.zeros(m)
    iteration = 0

    while np.linalg.norm(old_centroids - centroids) > 1e-15:
        old_centroids = centroids

        # for each point find closest centroid
        clusters_index = closest_centroid(centroids, X)

        # recompute centroids
        centroids = recompute_centroids(X, k, clusters_index)

        print('iteration=', iteration)
        iteration += 1

    draw_clusters(X, clusters_index, centroids, k)


if __name__ == '__main__':
    X = pd.read_csv('X.dat', sep='\s+', header=None).as_matrix()
    np.random.seed(42)
    k_means(X)
