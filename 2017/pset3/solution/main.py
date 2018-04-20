import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread

from solution.kmean import kmean, kmean_sklearn


def compress(image, centroids, num_clusters=16):
    dists_list = []
    X_large = image.reshape(-1, 3)
    dim = image.shape[0]

    for c in centroids:
        ds = np.sqrt(np.sum((X_large - c) ** 2, axis=1))
        dists_list.append(ds)
    assign = np.stack(dists_list).argmin(axis=0)

    image_compressed = np.zeros_like(X_large)
    for k in range(num_clusters):
        idxes = np.where(assign == k)[0]
        image_compressed[idxes] = centroids[k]

    return image_compressed.reshape(dim, dim, 3)


if __name__ == '__main__':
    im_large = imread('mandrill-large.tiff')
    im_small = imread('mandrill-small.tiff')

    X = im_small.reshape(-1, 3)
    # centroids, _, _ = kmean(X)
    centroids = kmean_sklearn(X)
    im_large_compressed = compress(im_large, centroids)

    plt.imshow(im_large_compressed)
    plt.show()
