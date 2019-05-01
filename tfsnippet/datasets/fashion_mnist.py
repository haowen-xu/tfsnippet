import gzip
import hashlib

import numpy as np
import idx2numpy

from tfsnippet.utils import CacheDir

__all__ = ['load_fashion_mnist']


TRAIN_X_URI = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
TRAIN_X_MD5 = '8d4fb7e6c68d591d4c3dfef9ec88bf0d'
TRAIN_Y_URI = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
TRAIN_Y_MD5 = '25c81989df183df01b3e8a0aad5dffbe'
TEST_X_URI = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
TEST_X_MD5 = 'bef4ecab320f06d8554ea6380940ec79'
TEST_Y_URI = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
TEST_Y_MD5 = 'bb300cfdad3c16e7a12a480ee83cd310'


def _fetch_array(uri, md5):
    """Fetch an MNIST array from the `uri` with cache."""
    path = CacheDir('fashion_mnist').download(
        uri, hasher=hashlib.md5(), expected_hash=md5)
    with gzip.open(path, 'rb') as f:
        return idx2numpy.convert_from_file(f)


def _validate_x_shape(x_shape):
    x_shape = tuple([int(v) for v in x_shape])
    if np.prod(x_shape) != 784:
        raise ValueError('`x_shape` does not product to 784: {!r}'.
                         format(x_shape))
    return x_shape


def load_fashion_mnist(x_shape=(28, 28), x_dtype=np.float32,
                       y_dtype=np.int32, normalize_x=False):
    """
    Load the Fashion MNIST dataset as NumPy arrays.

    Homepage: https://github.com/zalandoresearch/fashion-mnist

    Args:
        x_shape: Reshape each digit into this shape.  Default ``(784,)``.
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
