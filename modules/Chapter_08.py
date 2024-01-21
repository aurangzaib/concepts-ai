# System Libraries
from tensorflow import keras
import tensorflow as tf
import numpy as np

# User Libraries
from modules import Common


def dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train.reshape((60000, 28, 28, 1)), x_test.reshape((10000, 28, 28, 1))
    return Common.shuffle_data(x_train, y_train), Common.shuffle_data(x_test, y_test)


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
