import numpy as np
import matplotlib.pyplot as plt

from kmeans_sklearn import k_means_sklearn
from matplotlib.image import imread


def compress_sol(image_2D, centroids, num_clusters=16):
    dists_list = []
    dim = int(np.sqrt(image_2D.shape[0]))

    for c in centroids:
        ds = np.sqrt(np.sum((image_2D - c) ** 2, axis=1))
        dists_list.append(ds)
    assign = np.stack(dists_list).argmin(axis=0)

    image_compressed = np.zeros_like(image_2D)
    for k in range(num_clusters):
        idxes = np.where(assign == k)[0]
        image_compressed[idxes] = centroids[k]

    return image_compressed.reshape(dim, dim, 3)


def compress(image_2D, centroids, num_clusters=16):
    m, n = image_2D.shape
    dim = int(np.sqrt(m))

    image_2D_compressed = np.zeros_like(image_2D)
    for i in range(m):
        closests_centroid = centroids[(np.argmin(np.linalg.norm(centroids - image_2D[i, :], axis=1)))]
        image_2D_compressed[i, :] = closests_centroid

    return image_2D_compressed.reshape(dim, dim, 3)


if __name__ == '__main__':
    im_large_3D = imread('mandrill-large.tiff')
    im_large_2D = im_large_3D.reshape(-1, 3)

    print(im_large_3D.shape)
    print(im_large_2D.shape)

    # im_small_3D = imread('mandrill-small.tiff')
    # im_small_2D = im_small_3D.reshape(-1, 3)
    #
    # centroids = k_means_sklearn(im_small_2D)
    # image1 = compress_sol(im_large_2D, centroids)
    # image2 = compress(im_large_2D, centroids)
    #
    # # print(image1[0, :10, :])
    # # print(image2[0, :10, :])
    # # print(image1 == image2)
    #
    # plt.imshow(image2)
    # plt.show()
