import numpy as np


def normalize(value, minimum, maximum):

    return (value - minimum) / (maximum - minimum)


def clip(value, minimum, maximum):

    return np.clip(value, minimum, maximum)