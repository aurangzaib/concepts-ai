# System Libraries
from tensorflow import keras
import pathlib
import shutil
import os

# User Libraries
from modules import common


def dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[1], 1))
    return common.shuffle_data(x_train, y_train), common.shuffle_data(x_test, y_test)


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


def compile(model, model_path):
    model_dir = "../resources/models/cats_dogs/"
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
        optimizer=keras.optimizers.legacy.RMSprop(),
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.TensorBoard(log_dir="../resources/logs"),
        keras.callbacks.ModelCheckpoint(
            filepath=model_dir + "/" + model_path,
            monitor="val_binary_accuracy",
            save_best_only=True,
        ),
    ]
    return model, callbacks


def train_batch(train_dataset, val_dataset, test_dataset, model, model_path, epochs, batch_size):
    model, callbacks = compile(model, model_path)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=False,
        batch_size=batch_size,
    )
    model.evaluate(test_dataset)
    common.plot(data=[history], labels=[model_path])


def train(x_train, y_train, x_val, y_val, x_test, y_test, model, model_path, epochs, batch_size):
    model, callbacks = compile(model, model_path)
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=False,
        batch_size=batch_size,
    )
    model.evaluate(x_test, y_test)
    common.plot(data=[history], labels=[model_path])
