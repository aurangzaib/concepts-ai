import matplotlib.pylab as plt
from tensorflow import keras
from typing import Callable
import numpy as np
import functools
import time

"""
-------------------------------------------
Data Visualization Functions
-------------------------------------------
"""


def plot(data, labels, window_titles=["Accuracy", "Loss"], only_val=False, start_index=0):
    # Name of loss and accuracy metrics
    keys = history_keys(data[0])
    loss_name, acc_name = keys[0], keys[1]
    x = [value for value in range(len(data[0].history[acc_name]))]
    fig = plt.figure(figsize=(20, 5))
    val_acc_name = "val_{}".format(acc_name)
    val_loss_name = "val_{}".format(loss_name)

    axis1 = fig.add_subplot(1, 2, 1)
    axis2 = fig.add_subplot(1, 2, 2)
    for history, label in zip(data, labels):
        # ----------------------------------------------------------
        # Metric 1
        # ----------------------------------------------------------
        axis1.set_title(window_titles[0])
        # Training Graphs
        if only_val is False:
            axis1.plot(
                x[start_index:],
                history.history[acc_name][start_index:],
                label="{} - {}".format(window_titles[0], label),
            )
        # Validation Graphs
        axis1.plot(
            x[start_index:],
            history.history[val_acc_name][start_index:],
            label="Val {} - {}".format(window_titles[0], label),
        )
        # ----------------------------------------------------------
        # Metric 2
        # ----------------------------------------------------------
        axis2.set_title(window_titles[1])
        # Training Graphs
        if only_val is False:
            axis2.plot(
                x[start_index:],
                history.history[loss_name][start_index:],
                label="{} - {}".format(window_titles[1], label),
            )
        # Validation Graphs
        axis2.plot(
            x[start_index:],
            history.history[val_loss_name][start_index:],
            label="Val {} - {}".format(window_titles[1], label),
        )

    axis1.legend()
    axis1.grid()
    axis2.legend()
    axis2.grid()
    plt.show()


def plot_simple(data, labels=["Train Loss", "Val Loss"], start_index=0):
    x_axis = range(0, len(data[0]), 1)
    fig = plt.figure(figsize=(20, 5))
    axis = fig.add_subplot(1, 1, 1)
    for y, l in zip(data, labels):
        axis.plot(x_axis[start_index:], y[start_index:], label=l)
    axis.legend()
    axis.grid()
    plt.show()


def plot_img(img1, img2):
    fig = plt.figure(figsize=(20, 8))
    axis1 = fig.add_subplot(1, 2, 1)
    axis1.imshow(img1)
    axis2 = fig.add_subplot(1, 2, 2)
    axis2.imshow(img2)
    axis1.axis("off"), axis2.axis("off")
    plt.show()


"""
-------------------------------------------
Data Split Functions
-------------------------------------------
"""


@staticmethod
def split_data(samples, labels, percent=0.3):
    # Training and validation dataset
    num_val = int(percent * len(samples))
    x_train = samples[num_val:]
    y_train = labels[num_val:]
    x_val = samples[:num_val]
    y_val = labels[:num_val]
    # Dataset
    return x_train, y_train, x_val, y_val


@staticmethod
def split_data_kfold(data, fold_count, fold_size):
    """
    Note: np.vstack does not work when any array is empty
    Note: Data must be numpy arrays, not python array
    Note: Don't shuffle data between folds
    """
    val = data[fold_count * fold_size : (fold_count + 1) * fold_size]
    pre_fold = data[: fold_count * fold_size]
    post_fold = data[(fold_count + 1) * fold_size :]
    train = np.concatenate([pre_fold, post_fold], axis=0)
    return train, val


@staticmethod
def explore_k_fold():
    data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    n = 4
    size = len(data) // n
    for index in range(n):
        split_data_kfold(data, index, size)


def history_keys(history):
    print("Metrics: ")
    keys = []
    for key in history.history.keys():
        keys.append(key)
        print(key, end=", ")
    print()
    return keys


"""
-------------------------------------------
Data Augmentation Functions
-------------------------------------------
"""


def add_white_noise(input):
    print(input.shape)
    return np.hstack((input, np.random.random(size=(input.shape[0], input.shape[1]))))


def add_zero_channel(input):
    return np.hstack((input, np.zeros(shape=(input.shape[0], input.shape[1]))))


"""
-------------------------------------------
Data Manipulation Functions
-------------------------------------------
"""


def shuffle_data(input_samples, input_labels):
    shuffle_indices = np.random.permutation(len(input_samples))
    return input_samples[shuffle_indices], input_labels[shuffle_indices]


def multihot_encode_data(input_samples, dimension):
    """
    Samples:
    - Length of each review is not uniform
    - Use Multi-hot encoding to have a uniform review length
    """
    output_samples = np.zeros(shape=(input_samples.shape[0], dimension))
    for sample_index, word_indices in enumerate(input_samples):
        output_samples[sample_index][word_indices] = 1
    return output_samples


def key_value_swap(dataset):
    input = dataset.get_word_index()
    return dict([(value, key) for (key, value) in input.items()])


def valuekey_to_keyvalue(dataset, input: list):
    dictionary = key_value_swap(dataset)
    return [dictionary[i] for i in input]


"""
-------------------------------------------
Training Manipulation Functions
-------------------------------------------
"""


def callbacks(
    metric_stop="val_loss",
    metric_model="val_sparse_categorical_accuracy",
    model_dir="../resources/models/tmp/model.keras",
    log_dir="../resources/logs",
):
    return [
        keras.callbacks.EarlyStopping(monitor=metric_stop, patience=3),
        keras.callbacks.ModelCheckpoint(filepath=model_dir, monitor=metric_model, save_only_best=True),
        keras.callbacks.TensorBoard(log_dir=log_dir),
    ]


"""
-------------------------------------------
Helper Functions
-------------------------------------------
"""


def timer(func: Callable):
    @functools.wraps(func)  # Recommended: To retain function information
    def wrapper(*argv, **kwargs):
        t1 = time.perf_counter()  # Start time
        data = func(*argv, **kwargs)  # Call decorated function
        t2 = time.perf_counter()  # End time
        print("{}: {:.10f}ms".format(func.__name__, t2 - t1))
        return data  # Return result of decorated function

    return wrapper  # Return wrapper


def own_properties(instance):
    return [attr for attr in dir(instance) if not attr.startswith("_")]
