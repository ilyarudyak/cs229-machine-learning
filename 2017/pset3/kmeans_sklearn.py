from sklearn.cluster import KMeans
import pandas as pd

from draw_clusters import draw_clusters


def k_means_sklearn(X, k=16):
    kmeans = KMeans(n_clusters=k, max_iter=30).fit(X)
    # draw_clusters(X, kmeans.labels_, kmeans.cluster_centers_, k)
    return kmeans.cluster_centers_


if __name__ == '__main__':
    X = pd.read_csv('X.dat', sep='\s+', header=None).as_matrix()
    k_means_sklearn(X, 3)
