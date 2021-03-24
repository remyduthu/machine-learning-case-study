#!/usr/bin/env python3

from datetime import datetime
from numpy import array, concatenate, empty, ravel, reshape
from os import getcwd
from PIL import Image
from matplotlib import pyplot
from scipy.io import loadmat
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def flatten(X: list, y: list) -> (list, list):
    X = X
    y = y

    # Flatten labels & images
    y = ravel(y)

    samples, img_w, img_h = X.shape
    X = reshape(X, (samples, img_w * img_h))

    return X, y


def preprocess(images: list, labels: list, keep: int = None):
    if keep != None:
        images = images[:, :, :, :keep]
        labels = labels[:keep]

    img_width, img_height, _, img_count = images.shape
    processed_images = empty([img_count, img_width, img_height])

    for i in range(img_count):
        image = get_image(images, i)

        processed_images[i] = array(Image.fromarray(image).convert("L"))

    return flatten(processed_images, labels)


def get_image(images: list, index: int):
    return images[:, :, :, index]


def load_dataset(include_extra: bool = False):
    train_data = loadmat(f"{getcwd()}/data/train_32x32.mat")
    test_data = loadmat(f"{getcwd()}/data/test_32x32.mat")

    if include_extra:
        extra_data = loadmat(f"{getcwd()}/data/extra_32x32.mat")

        X_train = concatenate((train_data["X"], extra_data["X"]), axis=-1)
        y_train = concatenate((train_data["y"], extra_data["y"]))

        return X_train, y_train, test_data["X"], test_data["y"]

    return train_data["X"], train_data["y"], test_data["X"], test_data["y"]


def main():
    start_time = datetime.now()

    print("⚙️  Load the dataset...")
    X_train, y_train, X_test, y_test = load_dataset(include_extra=False)

    print("📐 Preprocess images...")

    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)

    print(f"🚧 Building the model (this may take some time)...")

    mlp = MLPClassifier(
        early_stopping=True,
        random_state=0,
        solver="adam",
        verbose=2,
    )

    clf = make_pipeline(
        StandardScaler(),
        mlp,
    )

    clf.fit(X_train, y_train)

    print(f"💡 Score: {clf.score(X_test, y_test) * 100}%")

    print(f"🕐 Elapsed time: {(datetime.now() - start_time).total_seconds()}sec")

    pyplot.plot(mlp.loss_curve_)
    pyplot.savefig("data/loss-curve.png")

    plot_confusion_matrix(clf, X_test, y_test)
    pyplot.savefig("data/confusion-matrix.png")


if __name__ == "__main__":
    main()
