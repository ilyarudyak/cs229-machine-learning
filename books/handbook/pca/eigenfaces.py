import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


def get_faces():
    return fetch_lfw_people(min_faces_per_person=60)


def get_components():
    pca = PCA(n_components=150, svd_solver='randomized')
    pca.fit(faces.data)
    return pca.components_, pca.explained_variance_ratio_


def show_components():
    _, axes = plt.subplots(3, 8, figsize=(9, 4))
    for c, ax in zip(components, axes.flat):
        ax.set_axis_off()
        ax.imshow(c.reshape(size), cmap='bone')
    plt.show()


def show_explained_variance():
    plt.plot(np.cumsum(variance))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


if __name__ == '__main__':
    sns.set()
    size = [62, 47]

    faces = get_faces()
    components, variance = get_components()
    # show_components()
    show_explained_variance()
