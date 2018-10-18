from functools import partial

import numpy as np
from tensorflow import keras as K

__all__ = ['load_mnist', 'load_cifar', 'load_cifar10', 'load_cifar100']


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


def load_cifar(labels='cifar10', channels_last=False, flatten=False, dtype=None,
               normalize=False):
    """
    Load the CIFAR dataset.

    Args:
        labels ({"cifar10", "cifar100"}): Specify the labels to load.
        channels_last (bool): If True, will place the channels in the last
            dimension.  Otherwise will place the channels in the second
            dimension. (default :obj:`False`)
        flatten (bool): If True, will flatten each image into 1-d array.
            This transformation will be applied after the `channels_last`
            transformation, if it is enabled. (default :obj:`False`)
        dtype: If specified, cast each digit into this dtype.
        normalize (bool): Whether or not to normalize the digits to (0, 1)?
            This implies ``dtype=np.float32``, if dtype is not specified.

    Returns:
        Tuple of numpy arrays `(x_train, y_train), (x_test, y_test)`.
    """
    if labels == 'cifar10':
        (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    elif labels == 'cifar100':
        (x_train, y_train), (x_test, y_test) = K.datasets.cifar100.load_data()
    else:
        raise ValueError('`labels` must be one of: {"cifar10", "cifar100"}.')
    assert(x_train.shape[1:] == (32, 32, 3))
    assert(y_train.shape[1:] == (1,))
    if not channels_last:
        axes_shuffle = [0, 3, 1, 2]
        x_train = np.transpose(x_train, axes_shuffle)
        x_test = np.transpose(x_test, axes_shuffle)
    if flatten:
        x_train = x_train.reshape([len(x_train), -1])
        x_test = x_test.reshape([len(x_test), -1])
    if dtype is None and normalize:
        dtype = np.float32
    if dtype is not None:
        x_train = x_train.astype(dtype=dtype)
        x_test = x_test.astype(dtype=dtype)
    if normalize:
        x_train /= 255.
        x_test /= 255.
    y_train = y_train.reshape([-1])
    y_test = y_test.reshape([-1])
    return (x_train, y_train), (x_test, y_test)


load_cifar10 = partial(load_cifar, 'cifar10')
load_cifar100 = partial(load_cifar, 'cifar100')
