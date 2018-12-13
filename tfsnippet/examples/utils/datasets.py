import gzip
from functools import partial

import idx2numpy
import numpy as np
from tensorflow import keras as K

from tfsnippet.dataflow import DataFlow
from tfsnippet.utils import CacheDir, DocInherit

__all__ = ['load_mnist', 'load_cifar', 'load_cifar10', 'load_cifar100']


@DocInherit
class Dataset(object):
    """Base class for a dataset."""

    def __init__(self, cache_name):
        """
        Construct a new :class:`MNIST` instance.

        Args:
            cache_name (str): The name of the caching directory.
        """
        self._cache_dir = CacheDir(cache_name)

    def train_arrays(self):
        """
        Get the training data as arrays.

        Returns:
            tuple[np.ndarray]: The training data arrays.
        """
        raise NotImplementedError()

    def test_arrays(self):
        """
        Get the testing data as arrays.

        Returns:
            tuple[np.ndarray]: The testing data arrays.
        """
        raise NotImplementedError()

    def train_flow(self, batch_size, shuffle=True, skip_incomplete=True,
                   random_state=None):
        """
        Get the training data as :class:`DataFlow`.

        Args:
            batch_size (int): Size of each mini-batch.
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).

        Returns:
            DataFlow: The training data flow.
        """
        return DataFlow.arrays(
            self.train_arrays(), batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state
        )

    def test_flow(self, batch_size, shuffle=False, skip_incomplete=False,
                  random_state=None):
        """
        Get the testing data as :class:`DataFlow`.

        Args:
            batch_size (int): Size of each mini-batch.
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).

        Returns:
            DataFlow: The testing data flow.
        """
        return DataFlow.arrays(
            self.test_arrays(), batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state,
        )


class MNIST(Dataset):
    """MNIST handwritten digits dataset."""

    def __init__(self, shape=None, dtype=None):
        """
        Construct a new :class:`MNIST` instance.

        Args:
            shape: If specified, reshape each digit into this shape.
            dtype: If specified, cast each digit into this dtype.
        """
        super(MNIST, self).__init__(cache_name='mnist')
        self._shape = shape
        self._dtype = dtype

    def _load_as_array(self, uri):
        path = self._cache_dir.download(uri)
        with gzip.open(path, 'rb') as f:
            return idx2numpy.convert_from_file(f)

    def _post_process_x(self, x):
        assert(x.shape[1:] == (28, 28))
        if self._shape is not None:
            x_shape = (-1,) + tuple(self._shape)
            x = x.reshape(x_shape)
        if self._dtype is not None:
            x = x.astype(self._dtype)
        return x

    def _post_process_y(self, y):
        assert(y.shape[1:] == ())
        return y

    def train_arrays(self):
        train_x = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        train_y = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        assert(len(train_x) == len(train_y))
        assert(len(train_x) == 60000)
        return self._post_process_x(train_x), self._post_process_y(train_y)

    def test_arrays(self):
        test_x = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
        test_y = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
        assert(len(test_x) == len(test_y))
        assert(len(test_x) == 10000)
        return self._post_process_x(test_x), self._post_process_y(test_y)


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
    (x_train, y_train) = MNIST().train_arrays()
    (x_test, y_test) = MNIST().test_arrays()
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
