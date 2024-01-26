from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../")
from modules import common


@dataclass
class Constant:
    input_dim = 2
    output_dim = 1
    epochs = 500
    n = 0.001


class chapter_03:
    def __init__(self):
        self.dataset()

    def dataset(self):
        n_samples = 1000

        # ---------------------------------------------------------------
        # Generate Dataset
        # ---------------------------------------------------------------

        pos_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1.0, 0.5], [0.5, 1.0]], size=n_samples)
        neg_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1.0, 0.5], [0.5, 1.0]], size=n_samples)
        pos_labels = np.ones(shape=(n_samples, 1))
        neg_labels = np.zeros(shape=(n_samples, 1))

        pos_test_samples = np.array([[4, 0], [5, -2], [2, -1]]).astype(np.float32)
        neg_test_samples = np.array([[0, 2], [1, 4], [2, 4]]).astype(np.float32)

        samples = np.vstack(tup=(pos_samples, neg_samples)).astype(np.float32)
        labels = np.vstack(tup=(pos_labels, neg_labels)).astype(np.float32)
        self.test_samples = np.vstack(tup=(pos_test_samples, neg_test_samples)).astype(np.float32)

        # ---------------------------------------------------------------
        # Shuffle Dataset
        # ---------------------------------------------------------------

        samples, labels = common.shuffle_data(samples, labels)

        # ---------------------------------------------------------------
        # Split Dataset
        # ---------------------------------------------------------------

        (self.x_train, self.y_train, self.x_val, self.y_val) = common.split_data(
            samples=samples,
            labels=labels,
            percent=0.3,
        )

    def explore(self):
        plt.scatter(x=self.x_train[:, 0], y=self.x_train[:, 1])
        plt.scatter(x=self.x_val[:, 0], y=self.x_val[:, 1])
