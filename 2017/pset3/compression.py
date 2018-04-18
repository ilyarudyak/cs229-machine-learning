import numpy as np

from show_image import *

from sklearn.cluster import KMeans
from matplotlib.image import imread


def get_image_2D(image):
    """
    how to modify image into 2D - see here:
    https://dzone.com/articles/cluster-image-with-k-means
    """
    x, y, z = image.shape
    return image.reshape(x * y, z)


def cluster_small(filename, k=16):
    image_2D = get_image_2D(imread(filename))
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300)
    kmeans.fit(image_2D)
    return kmeans


def compress_large(filename, centroids):
    image = imread(filename)
    x, y, z = image.shape

    image_2D = get_image_2D(image)
    m, n = image_2D.shape

    image_2D_compressed = np.zeros((m, n))
    for i in range(m):
        index = np.argmin(np.linalg.norm(centroids - image_2D[i, :], axis=1))
        image_2D_compressed[i, :] = centroids[index]

    return image_2D_compressed.reshape(x, y, z)


def recreate_small():
    """
    how to recreate image after clustering -
    see link above
    """
    filename_small = 'mandrill-small.tiff'

    image = imread(filename_small)
    x, y, z = image.shape

    kmeans = cluster_small(filename_small)
    image_recovered = kmeans.cluster_centers_[kmeans.labels_].reshape(x, y, z)

    show_image(image_recovered)


def compress_and_show():
    filename_small = 'mandrill-small.tiff'
    filename_large = 'mandrill-large.tiff'

    kmeans = cluster_small(filename_small)
    centroids = kmeans.cluster_centers_
    compressed_image = compress_large(filename_large, centroids)

    show_image(compressed_image)


if __name__ == '__main__':
    compress_and_show()
