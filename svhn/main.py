#!/usr/bin/env python3

from os import getcwd
from scipy.io import loadmat


def load_dataset():
    return loadmat(f"{getcwd()}/data/test_32x32.mat"), loadmat(
        f"{getcwd()}/data/test_32x32.mat"
    )


def main():
    train_data, test_data = load_dataset()


if __name__ == "__main__":
    main()
