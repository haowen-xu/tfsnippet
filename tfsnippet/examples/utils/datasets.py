import gzip
from functools import partial

import idx2numpy
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from tfsnippet.dataflow import DataFlow
from tfsnippet.utils import CacheDir, DocInherit

__all__ = ['MNIST', 'load_cifar', 'load_cifar10', 'load_cifar100']


def validate_enum_arg(name, value, choices):
    if value not in choices:
        raise ValueError('`{}` is required to be one of these values: {}'.
                         format(name, choices))
    return value


def make_dataflow_for_8bit_x(arrays, process, shape, dtype, tf_session,
                             batch_size, shuffle, skip_incomplete,
                             random_state):
    validate_enum_arg(
        'process', process, (None, 'normalize', 'binarize', 'bernoulli',
                             'uniform_noise', 'uniform_noise_normalize'))

    x = arrays[0]
    arrays = list(arrays[1:])
    original_random_state = random_state
    random_state = random_state or np.random

    # reshape x into desired shape
    if shape is not None:
        x = np.reshape(x, [-1] + list(shape))

    # maybe we should cast x
    if dtype is not None:
        cast_float_x = lambda x: np.asarray(x, dtype=dtype)
    else:
        cast_float_x = lambda x: np.asarray(x, dtype=np.float32)

    # define the samplers
    if tf_session is not None:
        x_in = tf.placeholder(dtype=tf.as_dtype(x.dtype),
                              shape=[None] + list(x.shape[1:]))

        with tf.device('/device:CPU:0'):
            with tf.name_scope('x_float', values=[x_in]):
                x_float = tf.cast(x_in, dtype=tf.float32)

            with tf.name_scope('sample_bernoulli_x', values=[x_float]):
                x_bernoulli_prob = x_float / tf.constant(255., dtype=tf.float32)
                x_bernoulli_out = tf.cast(
                    tf.less(
                        tf.random_uniform(shape=tf.shape(x_float), minval=0.,
                                          maxval=1., dtype=tf.float32),
                        x_bernoulli_prob
                    ),
                    dtype=tf.as_dtype(dtype or np.int32)
                )

            with tf.name_scope('sample_uniform_noise_x', values=[x_float]):
                x_uniform_out = x_float + tf.random_uniform(
                    shape=tf.shape(x_float), minval=0., maxval=1.,
                    dtype=tf.float32
                )
                x_uniform_normalize_out = \
                    x_uniform_out / tf.constant(256., dtype=tf.float32)
                if dtype is not None:
                    tf_dtype = tf.as_dtype(dtype)
                    if tf_dtype != tf.float32:
                        x_uniform_out = tf.cast(x_uniform_out, dtype=tf_dtype)
                        x_uniform_normalize_out = tf.cast(
                            x_uniform_normalize_out, dtype=tf_dtype)

        def binarize_sampler(x):
            return tf_session.run(x_bernoulli_out, feed_dict={x_in: x})

        def uniform_noise_sampler(x):
            return tf_session.run(x_uniform_out, feed_dict={x_in: x})

        def uniform_noise_normalize_sampler(x):
            return tf_session.run(x_uniform_normalize_out, feed_dict={x_in: x})

    else:
        def binarize_sampler(x):
            return np.asarray(
                random_state.random(size=x.shape) < (
                    x / np.array(255., dtype=np.float32)),
                dtype=dtype or np.int32
            )

        def uniform_noise_sampler(x):
            return cast_float_x(x + random_state.random(size=x.shape))

        def uniform_noise_normalize_sampler(x):
            return cast_float_x(
                (x + random_state.random(size=x.shape)) /
                np.array(256., dtype=np.float32)
            )

    # generate the x mapper and process the original x array
    x_mapper = None
    if process in 'uniform_noise':
        def x_mapper(batch_x, *args):
            return (uniform_noise_sampler(batch_x),) + args

    elif process in 'uniform_noise_normalize':
        def x_mapper(batch_x, *args):
            return (uniform_noise_normalize_sampler(batch_x),) + args

    elif process == 'bernoulli':
        def x_mapper(batch_x, *args):
            return (binarize_sampler(batch_x),) + args

    elif process == 'binarize':
        x = np.asarray(x >= .5, dtype=dtype or np.int32)

    elif process == 'normalize':
        x = cast_float_x(x / np.array(255., dtype=np.float32))

    else:
        x = cast_float_x(x)

    # compose the data flow
    df = DataFlow.arrays([x] + arrays, batch_size=batch_size, shuffle=shuffle,
                         skip_incomplete=skip_incomplete,
                         random_state=original_random_state)
    if x_mapper is not None:
        df = df.map(x_mapper)

    return df


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

        This method should return the original training arrays, without any
        processing.  To get the processed data according to arguments specified
        at construction, use :meth:`train_flow` instead.

        Returns:
            tuple[np.ndarray]: The training data arrays.
        """
        raise NotImplementedError()

    def test_arrays(self):
        """
        Get the testing data as arrays.

        This method should return the original testing arrays, without any
        processing.  To get the processed data according to arguments specified
        at construction, use :meth:`test_flow` instead.

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

    def __init__(self, process_x=None, shape=None, dtype=None,
                 pre_sample_test=True, tf_session=None):
        """
        Construct a new :class:`MNIST` instance.

        Args:
            process_x (None or str): Specify one of the following processing
                method to get processed data flows:

                * normalize: Normalize x into [0, 1] by dividing it with 255.
                * binarize: Transform x into 1 if x >=128, otherwise into 0.
                * bernoulli: Treat the normalized x as the probability of
                    a Bernoulli random variable, and in each epoch, sample
                    binarized x according to the probability.
                * uniform_noise: In each epoch, add a uniform noise
                    :math:`u \\sim U[0,1)` to x.
                * uniform_noise_normalize: In each epoch, add a uniform noise
                    :math:`u \\sim U[0,1)` to x, and then normalize the noisy
                    x into [0, 1] by dividing it with 256.

            shape: If specified, reshape each digit into this shape.
            dtype: If specified, cast each digit into this dtype.
            pre_sample_test (bool): Whether or not to pre-sample the testing
                flow instead of sampling at each epoch? (default :obj:`True`)
                This argument only takes effect if the `process_x` argument
                requires random sampling.
            tf_session (tf.Session or None): If specified, use TensorFlow
                sampler on this session.  Otherwise use NumPy sampler.
        """
        super(MNIST, self).__init__(cache_name='mnist')
        self._process = process_x
        self._shape = shape
        self._dtype = dtype
        self._pre_sample_test = pre_sample_test
        self._tf_session = tf_session

    def _load_as_array(self, uri):
        path = self._cache_dir.download(uri)
        with gzip.open(path, 'rb') as f:
            return idx2numpy.convert_from_file(f)

    def train_arrays(self):
        train_x = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        train_y = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        assert(len(train_x) == len(train_y))
        assert(len(train_x) == 60000)
        return train_x, train_y

    def test_arrays(self):
        test_x = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
        test_y = self._load_as_array(
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
        assert(len(test_x) == len(test_y))
        assert(len(test_x) == 10000)
        return test_x, test_y

    def train_flow(self, batch_size, shuffle=True, skip_incomplete=True,
                   random_state=None):
        return make_dataflow_for_8bit_x(
            self.train_arrays(), process=self._process, shape=self._shape,
            tf_session=self._tf_session,
            dtype=self._dtype, batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state
        )

    def test_flow(self, batch_size, shuffle=False, skip_incomplete=False,
                  random_state=None):
        df = make_dataflow_for_8bit_x(
            self.test_arrays(), process=self._process, shape=self._shape,
            tf_session=self._tf_session,
            dtype=self._dtype, batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state
        )
        if self._pre_sample_test:
            df = df.to_arrays_flow(batch_size=batch_size, shuffle=shuffle,
                                   skip_incomplete=skip_incomplete)
        return df


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
