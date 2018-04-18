import matplotlib.pyplot as plt


def draw_clusters(X, clusters_index, centroids, k=3):
    plt.clf()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(k):
        plt.plot(X[clusters_index == i, 0],
                 X[clusters_index == i, 1], colors[i] + '.')
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx')
    plt.show()
