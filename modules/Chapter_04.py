import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# =================================================================================
# Common
# =================================================================================
class Common:
    @staticmethod
    def key_value_swap(dataset):
        input = dataset.get_word_index()
        return dict([(value, key) for (key, value) in input.items()])

    """
    -------------------------------------------
    Data Manipulation Functions
    -------------------------------------------
    """

    @staticmethod
    def decode_data(dataset, input: list):
        dictionary = Common.key_value_swap(dataset)
        return [dictionary[i] for i in input]

    @staticmethod
    def encode_data(input_samples, dimension):
        """
        Samples:
        - Length of each review is not uniform
        - Use Multi-hot encoding to have a uniform review length
        """
        output_samples = np.zeros(shape=(len(input_samples), dimension))
        for sample_index, word_indices in enumerate(input_samples):
            output_samples[sample_index][word_indices] = 1
        return output_samples

    @staticmethod
    def shuffle_data(input_samples, input_labels):
        shuffle_indices = np.random.permutation(len(input_samples))
        return input_samples[shuffle_indices], input_labels[shuffle_indices]

    """
    -------------------------------------------
    Train/Val Split Functions
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
        Note: Data must be numpy arrays, not python array
        Note: Don't shuffle data between folds
        """
        val = data[fold_count * fold_size : (fold_count + 1) * fold_size]
        train = np.concatenate(
            [
                # Pre Validation Fold
                data[: fold_count * fold_size],
                # Post Validation Fold
                data[(fold_count + 1) * fold_size :],
            ],
            axis=0,
        )
        return train, val

    @staticmethod
    def explore_k_fold():
        data = np.array(
            [
                np.array([1, 2, 3]),
                np.array([4, 5, 6]),
                np.array([7, 8, 9]),
                np.array([10, 11, 12]),
                np.array([13, 14, 15]),
            ]
        )
        n = 4
        size = len(data) // n
        for index in range(n):
            Common.split_data_kfold(data, index, size)

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

    """
    -------------------------------------------
    Training Exploration Functions
    -------------------------------------------
    """

    def explore(train_loss, train_accuracy, val_loss, val_accuracy, with_text=True):
        plt.title("Training and validation loss")
        plt.plot(train_loss, "b", label="Training loss")
        plt.plot(val_loss, "r", label="Validation loss")
        plt.legend()
        plt.show()

        plt.title("Training and validation accuracy")
        plt.plot(train_accuracy, "b", label="Training accuracy")
        plt.plot(val_accuracy, "r", label="Validation accuracy")
        plt.legend()
        plt.show()

        if with_text is True:
            print("Training Loss: ", train_loss[-1])
            print("Training Accuracy: ", train_accuracy[-1])
            print("Validation Loss: ", val_loss[-1])
            print("Validation Accuracy: ", val_accuracy[-1])


# =================================================================================
# Binary Classification
# =================================================================================
class BinaryClassification:
    @staticmethod
    def prepare():
        num_words = 10000

        # Imdb dataset
        (x_train_og, y_train_og), (x_test_og, y_test_og) = BinaryClassification.dataset(
            num_words
        )

        # Shuffle dataset
        x_train_og, y_train_og = Common.shuffle_data(x_train_og, y_train_og)
        x_test_og, y_test_og = Common.shuffle_data(x_test_og, y_test_og)

        # Vectorize dataset
        (x_train, y_train), (x_test, y_test) = BinaryClassification.encode_data(
            x_train_og, y_train_og, x_test_og, y_test_og, num_words
        )

        # Split dataset
        x_train, y_train, x_val, y_val = Common.split_data(x_train, y_train, 0.4)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @staticmethod
    def dataset(num_words):
        imdb = tf.keras.datasets.imdb
        return imdb.load_data(num_words=num_words)

    def encode_data(x, y, xt, yt, num_words):
        x_train = Common.encode_data(x, num_words)
        y_train = np.asarray(y).astype(np.float32)
        x_test = Common.encode_data(xt, num_words)
        y_test = np.asarray(yt).astype(np.float32)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_model(learning_rate=0.001):
        # Create the model
        # Output: Probability
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=16, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=16, activation=tf.nn.relu),
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
                    units=16,
                    kernel_regularizer=regularizer.l2(weight_decay),
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.Dense(
                    units=16,
                    kernel_regularizer=regularizer.l2(weight_decay),
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

    def train(x, y, x_val, y_val, model, epochs):
        history = model.fit(
            x=x, y=y, epochs=epochs, validation_data=(x_val, y_val), verbose=False
        )
        return history

    @staticmethod
    def get_model_with_dropout(drop_rate=0.5, learning_rate=0.001):
        # Create the model
        # Output: Probability
        regularizer = tf.keras.regularizers
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=16, activation=tf.nn.relu),
                tf.keras.layers.Dropout(drop_rate),
                tf.keras.layers.Dense(units=16, activation=tf.nn.relu),
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

    def train(x, y, x_val, y_val, model, epochs):
        history = model.fit(
            x=x, y=y, epochs=epochs, validation_data=(x_val, y_val), verbose=False
        )
        return history

    @staticmethod
    def explore(history, with_text=True):
        train_loss = history.history["loss"]
        train_accuracy = history.history["binary_accuracy"]
        val_loss = history.history["val_loss"]
        val_accuracy = history.history["val_binary_accuracy"]

        Common.explore(train_loss, train_accuracy, val_loss, val_accuracy, with_text)


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
                tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
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
    def explore(history):
        print(history.history.keys())
        train_loss = history.history["loss"]
        train_accuracy = history.history["categorical_accuracy"]
        val_loss = history.history["val_loss"]
        val_accuracy = history.history["val_categorical_accuracy"]
        Common.explore(train_loss, train_accuracy, val_loss, val_accuracy)

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
        # Samples are centered arond 0
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
