import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def get_digits():
    digits = load_digits()
    return digits


def get_centroids(digits):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(digits.data)
    return kmeans.cluster_centers_.reshape(10, 8, 8)


def show_centroids(centroids):
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    for axi, center in zip(ax.flat, centroids):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap='binary')
    plt.show()


def get_labels(digits, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(digits.data)
    unordered_labels = kmeans.predict(digits.data)
    pred_labels = np.zeros_like(unordered_labels)
    true_labels = digits.target
    for label in range(n_clusters):
        mask = unordered_labels == label
        pred_labels[mask] = mode(true_labels[mask])[0]
    return pred_labels


def get_accuracy(digits, labels):

    # print(np.mean(digits.target == labels))
    return accuracy_score(digits.target, labels)


def get_confusion_matrix(digits, labels):
    return confusion_matrix(digits.target, labels)


def show_confusion_matrix(matrix):
    sns.heatmap(matrix.T, annot=True, fmt='d', cmap="YlGnBu",
                xticklabels=digits.target_names,
                yticklabels=digits.target_names
                )
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


if __name__ == '__main__':
    sns.set()

    digits = get_digits()
    labels = get_labels(digits)
    # print(f'accuracy={get_accuracy(digits, labels):.2}')
    matrix = get_confusion_matrix(digits, labels)
    show_confusion_matrix(matrix)