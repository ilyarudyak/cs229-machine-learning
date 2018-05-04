import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_sample_image


def get_image():
    return load_sample_image("china.jpg")


def show_image(image):
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(image)
    plt.show()


def reshape_image_to_kmeans(image, rgb=3):
    return image.reshape(-1, rgb)


def reshape_image_from_kmeans(image, h=427, w=640, rgb=3):
    return image.reshape(h, w, rgb)


def recolor_image(image, n_colors=16):
    image_reshaped = reshape_image_to_kmeans(image)

    kmeans = MiniBatchKMeans(n_clusters=n_colors).fit(image_reshaped)
    centroids = kmeans.cluster_centers_
    clusters = kmeans.predict(image_reshaped)

    for i in range(n_colors):
        image_reshaped[clusters == i, :] = centroids[i, :]

    return reshape_image_from_kmeans(image_reshaped)


if __name__ == '__main__':
    sns.set()

    china = get_image()
    china_recolored = recolor_image(china)
    show_image(china_recolored)
