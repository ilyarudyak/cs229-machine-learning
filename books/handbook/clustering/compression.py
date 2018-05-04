import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import MiniBatchKMeans
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

    # for i in range(n_colors):
    #     image_reshaped[clusters == i, :] = centroids[i, :]
    # return reshape_image_from_kmeans(image_reshaped)

    # vectorised implementation
    image_recolored = centroids[clusters].astype('uint8')

    # print(centroids[clusters[0]])
    # print(clusters[:10])
    # print(image_recolored[:10])
    return reshape_image_from_kmeans(image_recolored)


def show_images(images, titles):
    fig, ax = plt.subplots(1, images.shape[0], figsize=(16, 6))
    # fig.subplots_adjust(wspace=0.05)
    for axi, image, title in zip(ax.flat, images, titles):
        axi.set(xticks=[], yticks=[])
        axi.imshow(image)
        axi.set_title(title, size=16)
    plt.show()


if __name__ == '__main__':
    sns.set()

    china = get_image()
    china_recolored = recolor_image(china)
    images = np.array([china, china_recolored])
    titles = np.array(['original image', '16-color image'])
    show_images(images, titles)
