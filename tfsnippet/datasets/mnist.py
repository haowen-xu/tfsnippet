import gzip
import hashlib

import numpy as np
import idx2numpy

from tfsnippet.utils import CacheDir

__all__ = ['load_mnist']


TRAIN_X_URI = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_X_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
TRAIN_Y_URI = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TRAIN_Y_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'
TEST_X_URI = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_X_MD5 = '9fb629c4189551a2d022fa330f9573f3'
TEST_Y_URI = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
TEST_Y_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'


def _fetch_array(uri, md5):
    """Fetch an MNIST array from the `uri` with cache."""
    path = CacheDir('mnist').download(
        uri, hasher=hashlib.md5(), expected_hash=md5)
    with gzip.open(path, 'rb') as f:
        return idx2numpy.convert_from_file(f)


def _validate_x_shape(x_shape):
    x_shape = tuple([int(v) for v in x_shape])
    if np.prod(x_shape) != 784:
        raise ValueError('`x_shape` does not product to 784: {!r}'.
                         format(x_shape))
    return x_shape


def load_mnist(x_shape=(28, 28), x_dtype=np.float32, y_dtype=np.int32,
               normalize_x=False):
    """
    Load the MNIST dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.  Default ``(28, 28, 1)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
    """
    # check arguments
    x_shape = _validate_x_shape(x_shape)

    # load data
    train_x = _fetch_array(TRAIN_X_URI, TRAIN_X_MD5).astype(x_dtype)
    train_y = _fetch_array(TRAIN_Y_URI, TRAIN_Y_MD5).astype(y_dtype)
    test_x = _fetch_array(TEST_X_URI, TEST_X_MD5).astype(x_dtype)
    test_y = _fetch_array(TEST_Y_URI, TEST_Y_MD5).astype(y_dtype)

    assert(len(train_x) == len(train_y) == 60000)
    assert(len(test_x) == len(test_y) == 10000)

    # change shape
    train_x = train_x.reshape([len(train_x)] + list(x_shape))
    test_x = test_x.reshape([len(test_x)] + list(x_shape))

    # normalize x
    if normalize_x:
        train_x /= np.asarray(255., dtype=train_x.dtype)
        test_x /= np.asarray(255., dtype=test_x.dtype)

    return (train_x, train_y), (test_x, test_y)
