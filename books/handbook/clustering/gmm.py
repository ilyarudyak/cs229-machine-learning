import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def gen_data():
    X, y_true = make_blobs(n_samples=400, centers=n_clusters,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    return X, y_true


def show_data_sklearn(X):
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    clusters = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=clusters, s=20, cmap='tab10')
    plt.show()


def show_data_gmm(X):
    gmm = GaussianMixture(n_components=n_clusters).fit(X)
    clusters = gmm.predict(X)
    probs = gmm.predict_proba(X)
    print(probs[:10, :])

    size = 50 * probs.max(1) ** 2  # square emphasizes differences
    plt.scatter(X[:, 0], X[:, 1], c=clusters, s=size, cmap='tab10')
    plt.show()


if __name__ == '__main__':
    n_clusters = 4
    X, y_true = gen_data()
    show_data_gmm(X)
