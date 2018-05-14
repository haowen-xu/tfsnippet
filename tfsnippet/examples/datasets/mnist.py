import gzip
import os

import numpy as np

from .utils import *

__all__ = ['load_mnist']


def load_mnist(cache_dir=None, flatten_images=True, as_float=True, dtype=None,
               label_dtype=None):
    """
    Download mnist training and testing data as numpy array.

    Args:
        cache_dir (str): Path to the cache directory.  Will automatically
            choose one according to `get_cache_dir` if not specified.
        flatten_images (bool): If True, flatten images to 1D vectors.
            If False, shape the images to 3D tensors of shape (28, 28, 1),
            where the last dimension is the greyscale channel. (default True)
        as_float (bool): If True, scale the byte pixels to 0.0~1.0 float
            numbers. If False, keep the byte pixels in 0~255 byte numbers.
            (default True)
        dtype (numpy.dtype): Cast the image pixels into this type.
            If not specified, will use `np.float32` if `as_float`
            is True, or `numpy.uint8` if `as_float` is False.
        label_dtype (numpy.dtype): Cast the image labels into this type.
            If not specified, will use `numpy.int32`.

    Returns:
        ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
            Training set and testing set.
    """
    cache_dir = cache_dir or get_cache_dir('mnist')
    root_uri = 'http://yann.lecun.com/exdb/mnist/'

    def load_mnist_images(filename):
        cache_file = os.path.join(cache_dir, filename)
        cache_file = cached_download(root_uri + filename, cache_file)
        with gzip.open(cache_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        if flatten_images:
            data = data.reshape(-1, 784)
        else:
            data = data.reshape(-1, 28, 28, 1)

        if as_float:
            data = data / np.array(256, dtype=dtype or np.float32)
        elif dtype is not None:
            data = np.asarray(data, dtype=dtype)
        else:
            data = np.asarray(data, dtype=np.int32)

        return data

    def load_mnist_labels(filename):
        cache_file = os.path.join(cache_dir, filename)
        cache_file = cached_download(root_uri + filename, cache_file)
        with gzip.open(cache_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        if label_dtype is not None:
            data = data.astype(label_dtype)
        return data

    # We can now download and read the training and test set images and labels.
    train_X = load_mnist_images('train-images-idx3-ubyte.gz')
    train_y = load_mnist_labels('train-labels-idx1-ubyte.gz')
    test_X = load_mnist_images('t10k-images-idx3-ubyte.gz')
    test_y = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return (train_X, train_y), (test_X, test_y)
