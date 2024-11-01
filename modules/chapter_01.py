import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import sys

sys.path.append("../")
from modules import common


# =====================================================================
# Dataset
# =====================================================================


def prepare():
    # Mnist dataset
    (train_samples, train_labels), (test_samples, test_labels) = dataset(with_text=False)
    # Shuffle dataset
    train_samples, train_labels = common.shuffle_data(train_samples, train_labels)
    return (train_samples, train_labels), (test_samples, test_labels)


def dataset(with_text=True):
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if with_text is True:
        # Explore the dataset
        # Array of 60,000 images of uint8
        print("Data type: ", x_train.dtype)
        print("Samples: ", x_train.shape)
        print("Dim: ", x_train.ndim)
        visualize(x_test, y_test)

    # Flatten the dataset
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    # Rescale the dataset (0-255 to 0-1)
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    return (x_train, y_train), (x_test, y_test)


def get_model(input_size, num_classes):
    # -----------------------------------------
    # Input Layer
    # -----------------------------------------
    i = keras.Input(shape=input_size + (3,), name="Layer_Input")
    x = keras.layers.Rescaling(1.0 / 255, name="Layer_Rescale")(i)
    # -----------------------------------------
    # Downsampling Layers
    # -----------------------------------------
    x = keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu, strides=2, name="Conv_1")(x)
    x = keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu, name="Conv_2")(x)
    x = keras.layers.BatchNormalization()(x)
    r = x
    x = keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu, strides=2, name="Conv_3")(x)
    r = keras.layers.SeparableConv2D(filters=128, kernel_size=1, strides=2)(r)
    x = keras.layers.add([x, r])
    x = keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu, name="Conv_4")(x)
    x = keras.layers.SeparableConv2D(filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, strides=2, name="Conv_5")(x)
    x = keras.layers.SeparableConv2D(filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name="Conv_6")(x)
    # -----------------------------------------
    # Upsampling Layers
    # -----------------------------------------
    x = keras.layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, name="Conv_6T")(x)
    x = keras.layers.Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation=tf.nn.relu, strides=2, name="Conv_5T")(x)
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu, name="Conv_4T")(x)
    x = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation=tf.nn.relu, strides=2, name="Conv_3T")(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu, name="Conv_2T")(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu, strides=2, name="Conv_1T", kernel_regularizer=keras.regularizers.l2(l2=0.002))(x)
    x = keras.layers.Dropout(rate=0.25)(x)
    # -----------------------------------------
    # Output Layer
    # -----------------------------------------
    o = keras.layers.Conv2D(filters=num_classes, kernel_size=3, padding="same", activation=tf.nn.softmax, name="Layer_Output")(x)
    # -----------------------------------------
    # Model
    # -----------------------------------------
    model = keras.Model(inputs=i, outputs=o)
    return model


# =====================================================================
# Visualize
# =====================================================================


def visualize(x, y):
    plt.imshow(x[0], cmap=plt.cm.binary)
    print(y[0])
    plt.show()


# =====================================================================
# Model
# =====================================================================


def get_small_model(learning_rate=0.001):
    # ---------------------------------
    # Forward Propagation Configuration
    # ---------------------------------
    model = keras.Sequential(
        [
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    # ---------------------------------
    # Backward Propagation Configuration
    # ---------------------------------
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.legacy.RMSprop(learning_rate),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_model(learning_rate=0.001):
    # ---------------------------------
    # Forward Propagation Configuration
    # ---------------------------------
    model = keras.Sequential(
        [
            keras.layers.Dense(units=256, activation=tf.nn.relu),
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    # ---------------------------------
    # Backward Propagation Configuration
    # ---------------------------------
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.legacy.RMSprop(learning_rate),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_function_model(metrics=[keras.metrics.SparseCategoricalAccuracy()], with_compile=True):
    # ---------------------------------
    # Forward Propagation Configuration
    # ---------------------------------
    inputs = keras.layers.Input(shape=(28 * 28))
    features = keras.layers.Dense(units=256, activation=tf.nn.relu)(inputs)
    features = keras.layers.Dropout(0.01)(features)
    outputs = keras.layers.Dense(units=10, activation=tf.nn.softmax)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    # ---------------------------------
    # Backward Propagation Configuration
    # ---------------------------------
    if with_compile:
        model.compile(
            optimizer=keras.optimizers.legacy.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=metrics,
        )
    return model


# ---------------------------------
# Forward Propagation Configuration
# ---------------------------------
def get_large_model(learning_rate=0.001):
    model = keras.Sequential(
        [
            keras.layers.Dense(units=256, activation=tf.nn.relu),
            keras.layers.Dense(units=128, activation=tf.nn.relu),
            keras.layers.Dense(units=96, activation=tf.nn.relu),
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )
    compile(model, learning_rate=learning_rate)
    return model


# ---------------------------------
# Backward Propagation Configuration
# ---------------------------------
def compile(model, metrics=keras.metrics.SparseCategoricalAccuracy(), learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.legacy.RMSprop(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[metrics],
    )
    return model


# =====================================================================
# Train
# =====================================================================


def train(x, y, model, val_percent=0.3, batch_size=1024, callbacks=None, epochs=1):
    # Train the model
    history = model.fit(
        x=x,
        y=y,
        validation_split=val_percent,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=True,
        epochs=epochs,
    )
    return history


# =====================================================================
# Evaluate
# =====================================================================


def evaluate(x, y, model, silent=False):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x, y, verbose=False)
    if silent is False:
        print("Test Loss: ", test_loss)
        print("Test Acc: ", test_acc)


# =====================================================================
# Predict
# =====================================================================


def predict(x, y, model, silent=False):
    x_ground = x[:5]
    y_predict = model.predict(x_ground, verbose=False)
    if silent is False:
        for index in range(len(x_ground)):
            print()
            print("Ground: ", y[index])
            print("Prediction: ", (y_predict[index].argmax()))
            print("Confidence: ", (y_predict[index].max()))
