import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people


def get_faces():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print(faces.target_names)
    print(faces.images.shape)


if __name__ == '__main__':
    sns.set()
    get_faces()