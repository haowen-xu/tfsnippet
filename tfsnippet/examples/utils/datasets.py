import numpy as np
from tensorflow import keras as K

__all__ = ['load_mnist']


def load_mnist(shape=None, dtype=None, normalize=False):
    """
    Load the MNIST dataset.

    Args:
        shape: If specified, reshape each digit into this shape.
        dtype: If specified, cast each digit into this dtype.
        normalize (bool): Whether or not to normalize the digits to (0, 1)?
            This implies ``dtype=np.float32``, if dtype is not specified.

    Returns:
        Tuple of numpy arrays `(x_train, y_train), (x_test, y_test)`.
    """
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
    assert(x_train.shape[1:] == (28, 28))
    assert(y_train.shape[1:] == ())
    if shape is not None:
        x_shape = (-1,) + tuple(shape)
        x_train = x_train.reshape(x_shape)
        x_test = x_test.reshape(x_shape)
    if dtype is None and normalize:
        dtype = np.float32
    if dtype is not None:
        x_train = x_train.astype(dtype=dtype)
        x_test = x_test.astype(dtype=dtype)
    if normalize:
        x_train /= 255.
        x_test /= 255.
    return (x_train, y_train), (x_test, y_test)
