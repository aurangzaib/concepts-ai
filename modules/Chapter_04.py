import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

sys.path.append("../")
from modules import common


# =================================================================================
# Binary Classification
# =================================================================================
class BinaryClassification:
    @staticmethod
    def prepare():
        num_words = 10000

        # Imdb dataset
        (x_train_og, y_train_og), (x_test_og, y_test_og) = BinaryClassification.dataset(num_words)

        # Shuffle dataset
        x_train_og, y_train_og = common.shuffle_data(x_train_og, y_train_og)
        x_test_og, y_test_og = common.shuffle_data(x_test_og, y_test_og)

        # Vectorize dataset
        (x_train, y_train), (
            x_test,
            y_test,
        ) = BinaryClassification.multihot_encode_data(x_train_og, y_train_og, x_test_og, y_test_og, num_words)

        # Split dataset
        x_train, y_train, x_val, y_val = common.split_data(x_train, y_train, 0.4)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def dataset(num_words):
        imdb = tf.keras.datasets.imdb
        return imdb.load_data(num_words=num_words)

    def multihot_encode_data(x, y, xt, yt, num_words):
        x_train = common.multihot_encode_data(x, num_words)
        y_train = np.asarray(y).astype(np.float32)
        x_test = common.multihot_encode_data(xt, num_words)
        y_test = np.asarray(yt).astype(np.float32)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_model(learning_rate=0.001):
        # Create the model
        # Output: Probability
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid),
            ]
        )
        # Configure the model
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate),
        )
        return model

    @staticmethod
    def get_large_model(learning_rate=0.001):
        # Create the model
        # Output: Probability
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid),
            ]
        )
        # Configure the model
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate),
        )
        return model

    @staticmethod
    def get_model_with_regularization(weight_decay=0.002, learning_rate=0.001):
        # Create the model
        # Output: Probability
        regularizer = tf.keras.regularizers
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=1,
                    kernel_regularizer=regularizer.l2(l2=weight_decay),
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid),
            ]
        )
        # Configure the model
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate),
        )
        return model

    @staticmethod
    def get_model_with_dropout(drop_rate=0.1, learning_rate=0.001):
        # Create the model
        # Output: Probability
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=1, activation=tf.nn.relu),
                tf.keras.layers.Dropout(drop_rate),
                tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid),
            ]
        )
        # Configure the model
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate),
        )
        return model

    def train(x, y, x_val, y_val, model, epochs, batch_size=1024):
        history = model.fit(
            x=x,
            y=y,
            epochs=epochs,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            verbose=False,
        )
        return history


# =================================================================================
# Multiclass Classification
# =================================================================================
class MultiClassification:
    @staticmethod
    def dataset(num_words, reuters):
        return reuters.load_data(num_words=num_words)

    @staticmethod
    def get_model(learning_rate=0.001):
        # Create the model
        # Output: Probability Distribution
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(units=46, activation=tf.nn.softmax),
            ]
        )

        # Configure the model
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate),
        )
        return model

    @staticmethod
    def evaluate(x, y, model):
        evaluation = model.evaluate(x=x, y=y, verbose=False)
        print("Evaluation Loss: ", evaluation[0])
        print("Evaluation Accuracy: ", evaluation[1])

    @staticmethod
    def predict(x, y, model):
        y_predict = model.predict(x)
        for predict, ground in zip(y_predict[:5], y[:5]):
            print("----------------------------------------------")
            print("True: {}".format(np.argmax(ground)))
            print("Pred: {}".format(np.argmax(predict)))
            print("Confidence: {}%".format(int(max(predict) * 100)))


# =================================================================================
# Scalar Regression
# =================================================================================
class ScalarRegression:
    @staticmethod
    def dataset():
        return tf.keras.datasets.boston_housing.load_data()

    @staticmethod
    def normalize(train, test):
        # Samples are centered around 0
        mean = train.mean(axis=0)
        train -= mean

        # Samples have unit standard deviation
        std = train.std(axis=0)
        train /= std

        # Test dataset (using mean/std from samples)
        test -= mean
        test /= std

        return train, test

    @staticmethod
    def get_model(learning_rate=0.001):
        # Output: Continuous Value
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=96, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=96, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=1, activation=None),
            ]
        )

        # Configure the model
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate),
        )
        return model

    @staticmethod
    def get_prediction(x, y, model, type, num_predict=3):
        y_predict = model.predict(x, verbose=False)
        print()
        print("-------------------------------------------")
        print(type)
        print("-------------------------------------------")
        print()
        for predict, ground in zip(y_predict[:num_predict], y[:num_predict]):
            print("Ground: {} Predict: {}".format(ground, predict[0]))
            print("Error: {}".format(abs(ground - predict[0])))
            print()

    @staticmethod
    def get_evaluation(x, y, model, type):
        _, test_mae_score = model.evaluate(x, y, verbose=False)
        print("Mean Test MAE ({}): {}".format(type, test_mae_score))

    """
    -------------------------------------------
    History Functions
    -------------------------------------------
    """

    @staticmethod
    def save_history(histories, history):
        """
        histories: History of all epochs of all folds
        history:   History of current epoch of current fold
        """
        histories["loss"].append(history["loss"])
        histories["val_loss"].append(history["val_loss"])
        histories["mean_absolute_error"].append(history["mean_absolute_error"])
        histories["val_mean_absolute_error"].append(history["val_mean_absolute_error"])

    @staticmethod
    def get_history(history, type, num_ignore=20):
        # MAE per epoch
        mae = history["mean_absolute_error"][num_ignore:]
        val_mae = history["val_mean_absolute_error"][num_ignore:]
        print("Mean Training MAE: ", np.mean(mae))
        print("Mean Validation MAE: ", np.mean(val_mae))
        plt.title("MAE per epoch: {}".format(type))
        plt.plot(mae, "b", label="MAE Training")
        plt.plot(val_mae, "r", label="MAE Validation")
        plt.legend()
        plt.show()

    @staticmethod
    def mean_history(histories):
        """
        Function:
        -------------------------------------------------------------------
        - Mean of all folds per epoch

        Example:
        -------------------------------------------------------------------
        - Loss = [
        [[Epoch1_Fold1], [Epoch1_Fold2], [Epoch1_Fold3], [Epoch1_Fold4]],
        [[Epoch2_Fold1], [Epoch2_Fold2], [Epoch2_Fold3], [Epoch2_Fold4]],
        ...
        ]

        Note:
        -------------------------------------------------------------------
        - Dataset needs to be transposed. Because:
            Given:    Metric per fold per epoch
            Required: Metric per epoch per fold
        - First fold is ignored because its results are always unstable (reason not known)
        """
        history = {
            "loss": [],
            "val_loss": [],
            "mean_absolute_error": [],
            "val_mean_absolute_error": [],
        }
        m1 = np.array(histories["loss"][1:]).transpose()
        m2 = np.array(histories["val_loss"][1:]).transpose()
        m3 = np.array(histories["mean_absolute_error"][1:]).transpose()
        m4 = np.array(histories["val_mean_absolute_error"][1:]).transpose()
        history = {
            "loss": [np.mean(loss_per_fold) for loss_per_fold in m1],
            "val_loss": [np.mean(loss_per_fold) for loss_per_fold in m2],
            "mean_absolute_error": [np.mean(loss_per_fold) for loss_per_fold in m3],
            "val_mean_absolute_error": [np.mean(loss_per_fold) for loss_per_fold in m4],
        }
        return history
