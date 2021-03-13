#!/usr/bin/env python3

from os import getcwd
from scipy.io import loadmat


def get_image(images: list, index: int):
    return images[:, :, :, index]


def get_label(labels: list, index: int):
    return labels[index]


def load_dataset():
    return loadmat(f"{getcwd()}/data/test_32x32.mat"), loadmat(
        f"{getcwd()}/data/test_32x32.mat"
    )


def main():
    train_data, test_data = load_dataset()

    train_images = train_data["X"]
    train_labels = train_data["y"]

    train_image = get_image(train_images, 5)
    train_label = get_label(train_labels, 5)

    print(train_image.shape)
    print(train_label)

    test_images = test_data["X"]
    test_labels = test_data["y"]

    test_image = get_image(test_images, 5)
    test_label = get_label(test_labels, 5)

    print(test_image.shape)
    print(test_label)


if __name__ == "__main__":
    main()
