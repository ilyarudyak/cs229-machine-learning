import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


def get_faces():
    return fetch_lfw_people(min_faces_per_person=60)


def get_pca() -> PCA:
    pca = PCA(n_components=150, svd_solver='randomized')
    pca.fit(faces.data)
    return pca


def show_components():
    _, axes = plt.subplots(3, 8, figsize=(9, 4))
    for c, ax in zip(pca.components_, axes.flat):
        ax.set_axis_off()
        ax.imshow(c.reshape(size), cmap='bone')
    plt.show()


def show_explained_variance():
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def show_reconstructed_images():
    faces_reconstructed = pca.inverse_transform(pca.transform(faces.data))
    row, col = 2, 10
    _, axes = plt.subplots(row, col, figsize=(10, 3),
                           subplot_kw={'xticks': [], 'yticks': []},
                           gridspec_kw=dict(hspace=0.1, wspace=0.1)
                           )
    for j in range(col):
        axes[0, j].imshow(faces.images[j], cmap='binary_r')
        axes[1, j].imshow(faces_reconstructed[j].reshape(size), cmap='binary_r')

    axes[0, 0].set_ylabel('full-dim\ninput')
    axes[1, 0].set_ylabel('150-dim\nreconstruction')
    plt.show()


if __name__ == '__main__':
    sns.set()
    size = [62, 47]
    faces = get_faces()
    pca = get_pca()

    # show_components()
    # show_explained_variance()
    show_reconstructed_images()
