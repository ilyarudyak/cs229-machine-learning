from matplotlib.image import imread
import matplotlib.pyplot as plt


def show_image(image_filenames):
    number_images = len(image_filenames)
    for i in range(number_images):
        plt.subplot(1, number_images, i + 1)
        image = imread(image_filenames[i])
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    image_filenames = ['mandrill-large.tiff', 'mandrill-small.tiff']
    show_image(image_filenames)
