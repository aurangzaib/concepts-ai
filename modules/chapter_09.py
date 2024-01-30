# System Libraries
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import os

from modules import common


load_img = tf.keras.utils.load_img
img_to_array = tf.keras.utils.img_to_array


def get_path():
    input_dir = "../resources/datasets/02-oxford-pets/images/"
    target_dir = "../resources/datasets/02-oxford-pets/annotations/trimaps/"
    input_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
    label_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    return input_paths, label_paths, len(input_paths)


def explore(img, label):
    # Convert maps from (1,2,3) to (0, 127, 254)
    normalized_array = label * 127
    common.plot_img(img, normalized_array[:, :, 0])


def explore_test(input, pred):
    mask = np.argmax(pred, axis=-1) * 127
    common.plot_img(input, mask)


def get_inputs(path, img_size):
    return img_to_array(load_img(path, target_size=img_size))


def get_labels(path, img_size):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img


def dataset(img_size=(200, 200)):
    # Path of the images and labels
    input_paths, label_paths, num_labels = get_path()
    # Shuffle the paths
    random.Random(1337).shuffle(input_paths)
    random.Random(1337).shuffle(label_paths)
    # Image and label as array
    input_imgs = np.zeros((num_labels,) + img_size + (3,), dtype=np.float32)
    labels = np.zeros((num_labels,) + img_size + (1,), dtype=np.uint8)
    # Image as pixel
    input_raw = []
    for i in range(num_labels):
        input_raw.append(load_img(input_paths[i], target_size=img_size))
        input_imgs[i] = get_inputs(input_paths[i], img_size)
        labels[i] = get_labels(label_paths[i], img_size)
    return input_raw, input_imgs, labels, num_labels
