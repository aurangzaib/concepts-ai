# System Libraries
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pathlib
import shutil
import os

# User Libraries
from modules import Common


def dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train.reshape((60000, 28, 28, 1)), x_test.reshape((10000, 28, 28, 1))
    return Common.shuffle_data(x_train, y_train), Common.shuffle_data(x_test, y_test)


def dataset_batches(batch_size=1024):
    """
    1. Decode JPG to RGB matrices
    2. Convert RGB matrices to float tensors
    3. Resize to 180x180
    4. Batches of 32 images

    keras.utils.image_dataset_from_directory:
    1. Return batches of 32, 180, 180, 3
    """
    original_dir = pathlib.Path("../resources/datasets/cats_vs_dogs/")
    new_base_dir = pathlib.Path("../resources/datasets/cats_vs_dogs_small/")
    make_subsets(original_dir, new_base_dir)
    train_dataset = keras.utils.image_dataset_from_directory(
        new_base_dir / "train",
        image_size=(180, 180),
        batch_size=batch_size,
    )
    val_dataset = keras.utils.image_dataset_from_directory(
        new_base_dir / "validation",
        image_size=(180, 180),
        batch_size=batch_size,
    )
    test_dataset = keras.utils.image_dataset_from_directory(
        new_base_dir / "test",
        image_size=(180, 180),
        batch_size=batch_size,
    )
    return new_base_dir, train_dataset, val_dataset, test_dataset


def make_subset(subset_name, start_index, end_index, original_dir, new_base_dir):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)


def make_subsets(original_dir, new_base_dir):
    make_subset("train", 0, 1000, original_dir, new_base_dir)
    make_subset("validation", 1000, 1500, original_dir, new_base_dir)
    make_subset("test", 1500, 2500, original_dir, new_base_dir)


def get_model():
    return get_model_with_downsampling()


def get_model_without_downsampling():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)(inputs)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_model_with_downsampling():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)(inputs)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_model_with_padding():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)(inputs)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
