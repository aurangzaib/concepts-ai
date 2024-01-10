import numpy as np

def add_white_noise(input):
    print(input.shape)
    return np.concatenate(
        [
            input,
            np.random.random(size=(input.shape[0], input.shape[1])),
        ],
        axis=1,
    )


def add_zero_channel(input):
    return np.concatenate(
        [input, np.zeros(shape=(input.shape[0], input.shape[1]))],
        axis=1,
    )
