from sklearn.cluster import KMeans
import pandas as pd

from draw_clusters import draw_clusters


def k_means_sklearn(X, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=10).fit(X)
    draw_clusters(X, kmeans.labels_, kmeans.cluster_centers_, k)


if __name__ == '__main__':
    X = pd.read_csv('X.dat', sep='\s+', header=None).as_matrix()
    k_means_sklearn(X)
