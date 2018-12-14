import gzip

import numpy as np
import idx2numpy

from tfsnippet.dataflow import DataFlow
from tfsnippet.preprocessing import BernoulliSampler, UniformNoiseSampler
from tfsnippet.utils import CacheDir

__all__ = []


TRAIN_X_URI = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_Y_URI = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TEST_X_URI = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_Y_URI = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def _fetch_array(uri):
    """Fetch an MNIST array from the `uri` with cache."""
    path = CacheDir('mnist').download(uri)
    with gzip.open(path, 'rb') as f:
        return idx2numpy.convert_from_file(f)


def load(x_shape=(784,), x_dtype=np.float32, y_dtype=np.int32):
    """
    Load the MNIST dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.  Default ``(784,)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
    """
    train_x = _fetch_array(TRAIN_X_URI).astype(x_dtype)
    train_y = _fetch_array(TRAIN_Y_URI).astype(y_dtype)
    test_x = _fetch_array(TEST_X_URI).astype(x_dtype)
    test_y = _fetch_array(TEST_Y_URI).astype(y_dtype)

    assert(len(train_x) == len(train_y) == 60000)
    assert(len(test_x) == len(test_y) == 10000)

    train_x = train_x.reshape([-1] + list(x_shape))
    test_x = test_x.reshape([-1] + list(x_shape))
    return (train_x, train_y), (test_x, test_y)


def bernoulli_flow(x, batch_size, shuffle=False, skip_incomplete=False,
                   sample_now=False, dtype=np.int32, random_state=None):
    """
    Construct a new :class:`DataFlow`, which samples 0/1 binary images
    according to the given `x` array.

    Args:
        x: The `train_x` or `test_x` of MNIST dataset.
        batch_size (int): Size of each mini-batch.
        shuffle (bool): Whether or not to shuffle data before iterating?
            (default :obj:`False`)
        skip_incomplete (bool): Whether or not to exclude the last
            mini-batch if it is incomplete? (default :obj:`False`)
        sample_now (bool): Whether or not to sample immediately instead
            of sampling at the beginning of each epoch? (default :obj:`False`)
        dtype: The data type of the sampled array.  Default `np.int32`.
        random_state (RandomState): Optional numpy RandomState for
            shuffling data before each epoch.  (default :obj:`None`,
            use the global :class:`RandomState`).

    Returns:
        DataFlow: The Bernoulli `x` flow.
    """
    x = np.asarray(x)

    # prepare the sampler
    x = x / np.asarray(255., dtype=x.dtype)
    sampler = BernoulliSampler(dtype=dtype, random_state=random_state)

    # compose the data flow
    if sample_now:
        x = sampler(x)[0]
    df = DataFlow.arrays([x],
                         batch_size=batch_size,
                         shuffle=shuffle,
                         skip_incomplete=skip_incomplete,
                         random_state=random_state)
    if not sample_now:
        df = df.map(sampler)

    return df


def quantized_flow(x, batch_size, shuffle=False, skip_incomplete=False,
                   normalize=False, sample_now=False, dtype=np.float32,
                   random_state=None):
    """
    Construct a new :class:`DataFlow`, which adds uniform noises onto
    the given `x` array.

    Args:
        x: The `train_x` or `test_x` of MNIST dataset.
        batch_size (int): Size of each mini-batch.
        shuffle (bool): Whether or not to shuffle data before iterating?
            (default :obj:`False`)
        skip_incomplete (bool): Whether or not to exclude the last
            mini-batch if it is incomplete? (default :obj:`False`)
        normalize (bool): Whether or not to normalize the sampled array?
            If :obj:`True`, the sampled array would range in ``[0, 1)``.
            If :obj:`True`, the sampled array would range in ``[0, 256)``.
            Default :obj:`True`.
        sample_now (bool): Whether or not to sample immediately instead
            of sampling at the beginning of each epoch? (default :obj:`False`)
        dtype: The data type of the sampled array.  Default `np.float32`.
        random_state (RandomState): Optional numpy RandomState for
            shuffling data before each epoch.  (default :obj:`None`,
            use the global :class:`RandomState`).

    Returns:
        DataFlow: The quantized `x` flow.
    """
    x = np.asarray(x)

    # prepare the sampler
    if normalize:
        x = x / np.asarray(256., dtype=x.dtype)
        maxval = np.asarray(1 / 256., dtype=x.dtype)
    else:
        maxval = np.asarray(1., dtype=x.dtype)
    minval = np.asarray(0., dtype=x.dtype)
    sampler = UniformNoiseSampler(minval=minval, maxval=maxval, dtype=dtype,
                                  random_state=random_state)

    # compose the data flow
    if sample_now:
        x = x + sampler(x)[0]
    df = DataFlow.arrays([x],
                         batch_size=batch_size,
                         shuffle=shuffle,
                         skip_incomplete=skip_incomplete,
                         random_state=random_state)
    if not sample_now:
        df = df.map(sampler)

    return df
