import numpy as np

from sklearn.cluster import KMeans


def kmean(X, num_clusters=16):
    # picking random data points from the data as the
    # initial centroids to avoid empty cluster
    _idxes = np.random.choice(np.arange(X.shape[0]), size=num_clusters, replace=False)
    centroids = X[_idxes]

    err_history = []
    err = 1e6
    while err > 1:
        dists_list = []
        for c in centroids:
            ds = np.sqrt(np.sum((X - c) ** 2, axis=1))
            dists_list.append(ds)

        assign = np.stack(dists_list).argmin(axis=0)

        # new centroids
        nc_list = []
        for k in range(num_clusters):
            idxes = X[np.where(assign == k)[0]]
            nc_list.append(X[np.where(assign == k)[0]].mean(axis=0))

        nc = np.stack(nc_list)
        err = np.sum(np.abs(nc - centroids))
        err_history.append(err)
        centroids = nc

    return centroids, assign, err_history


def kmean_sklearn(X, num_clusters=16):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=30).fit(X)
    return kmeans.cluster_centers_
