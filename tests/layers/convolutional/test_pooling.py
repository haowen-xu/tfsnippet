import functools

import numpy as np
import tensorflow as tf
from mock import mock

from tfsnippet.layers import *
from tfsnippet.ops import flatten_to_ndims, unflatten_from_ndims
from tfsnippet.utils import is_integer


def patched_pool(pool_fn, value, ksize, strides, padding, data_format):
    """A patched version of `tf.nn.?_pool`, which emulates NCHW by NHWC."""
    if data_format == 'NCHW':
        transpose_axis = (0, 2, 3, 1)
        value = tf.transpose(value, transpose_axis)
        strides = tuple(strides[i] for i in transpose_axis)
        ksize = tuple(ksize[i] for i in transpose_axis)
    output = pool_fn(
        value=value, ksize=ksize, strides=strides, padding=padding,
        data_format='NHWC'
    )
    if data_format == 'NCHW':
        transpose_axis = (0, 3, 1, 2)
        output = tf.transpose(output, transpose_axis)
    return output


patched_avg_pool = functools.partial(patched_pool, tf.nn.avg_pool)
patched_max_pool = functools.partial(patched_pool, tf.nn.max_pool)


class Pooling2DTestCase(tf.test.TestCase):

    @staticmethod
    def pool2d_ans(pool_fn, input, pool_size, padding, strides):
        """Produce the expected answer of ?_pool2d."""
        strides = (strides,) * 2 if is_integer(strides) else tuple(strides)
        strides = (1,) + strides + (1,)
        ksize = (pool_size,) * 2 if is_integer(pool_size) else tuple(pool_size)
        ksize = (1,) + ksize + (1,)

        session = tf.get_default_session()
        input, s1, s2 = flatten_to_ndims(input, 4)
        padding = padding.upper()

        output = pool_fn(
            value=input,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format='NHWC',
        )

        output = unflatten_from_ndims(output, s1, s2)
        output = session.run(output)
        return output

    @staticmethod
    def run_pool2d(pool_fn, input, pool_size, padding, strides, channels_last,
                   ph=None, **kwargs):
        """Run `tfsnippet.layers.?_pool2d` and get the output."""
        i_shape = input.shape
        if not channels_last:
            input = np.transpose(
                input,
                tuple(i for i in range(len(i_shape) - 3)) + (-1, -3, -2)
            )

        session = tf.get_default_session()
        output = session.run(
            pool_fn(
                input=ph if ph is not None else input,
                pool_size=pool_size,
                channels_last=channels_last,
                padding=padding,
                strides=strides,
                **kwargs
            ),
            feed_dict={ph: input} if ph is not None else None
        )
        if not channels_last:
            output = np.transpose(
                output,
                tuple(i for i in range(len(i_shape) - 3)) + (-2, -1, -3)
            )
        return output

    def test_avg_pool(self):
        pool_fn = tf.nn.avg_pool
        with mock.patch('tensorflow.nn.avg_pool', patched_avg_pool), \
                self.test_session() as sess:
            np.random.seed(1234)
            x = np.random.normal(size=[17, 11, 32, 32, 5]).astype(np.float32)

            # test pool_size 1, strides 1, same padding, NHWC
            np.testing.assert_allclose(
                self.run_pool2d(avg_pool2d, x, 1, 'same', 1, True),
                self.pool2d_ans(pool_fn, x, 1, 'same', 1)
            )
            # test pool_size (3, 4), strides (2, 3), valid padding, NHWC
            np.testing.assert_allclose(
                self.run_pool2d(avg_pool2d, x, (3, 4), 'valid', (2, 3), True),
                self.pool2d_ans(pool_fn, x, (3, 4), 'valid', (2, 3))
            )
            # test pool_size (3, 4), strides (2, 3), valid padding, NCHW
            np.testing.assert_allclose(
                self.run_pool2d(avg_pool2d, x, (3, 4), 'valid', (2, 3), False),
                self.pool2d_ans(pool_fn, x, (3, 4), 'valid', (2, 3))
            )

    def test_max_pool(self):
        pool_fn = tf.nn.max_pool
        with mock.patch('tensorflow.nn.max_pool', patched_max_pool), \
                self.test_session() as sess:
            np.random.seed(1234)
            x = np.random.normal(size=[17, 11, 32, 32, 5]).astype(np.float32)

            # test pool_size 1, strides 1, same padding, NHWC
            np.testing.assert_allclose(
                self.run_pool2d(max_pool2d, x, 1, 'same', 1, True),
                self.pool2d_ans(pool_fn, x, 1, 'same', 1)
            )
            # test pool_size (3, 4), strides (2, 3), valid padding, NHWC
            np.testing.assert_allclose(
                self.run_pool2d(max_pool2d, x, (3, 4), 'valid', (2, 3), True),
                self.pool2d_ans(pool_fn, x, (3, 4), 'valid', (2, 3))
            )
            # test pool_size (3, 4), strides (2, 3), valid padding, NCHW
            np.testing.assert_allclose(
                self.run_pool2d(max_pool2d, x, (3, 4), 'valid', (2, 3), False),
                self.pool2d_ans(pool_fn, x, (3, 4), 'valid', (2, 3))
            )

    def test_global_avg_pool(self):
        def f(input, channels_last, keepdims, ph=None):
            i_shape = input.shape
            if not channels_last:
                input = np.transpose(
                    input,
                    tuple(i for i in range(len(i_shape) - 3)) + (-1, -3, -2)
                )

            session = tf.get_default_session()
            output = session.run(
                global_avg_pool2d(
                    input=ph if ph is not None else input,
                    channels_last=channels_last,
                    keepdims=keepdims
                ),
                feed_dict={ph: input} if ph is not None else None
            )
            if keepdims and not channels_last:
                output = np.transpose(
                    output,
                    tuple(i for i in range(len(i_shape) - 3)) + (-2, -1, -3)
                )
            return output

        with self.test_session() as sess:
            assert_allclose = functools.partial(
                np.testing.assert_allclose, atol=1e-6, rtol=1e-5)

            np.random.seed(1234)
            x = np.random.normal(size=[17, 11, 32, 32, 5]).astype(np.float32)
            a1 = np.mean(x, axis=(-3, -2), keepdims=True)
            a2 = np.mean(x, axis=(-3, -2), keepdims=False)

            # test NHWC
            assert_allclose(f(x, channels_last=True, keepdims=True), a1)
            # test NCHW, not keep dims
            assert_allclose(f(x, channels_last=False, keepdims=False), a2)
            # test dynamic dimensions, NHWC, not keep dims
            ph = tf.placeholder(
                dtype=tf.float32, shape=(None, None, None, None, 5))
            assert_allclose(f(x, channels_last=True, keepdims=False, ph=ph), a2)
            # test dynamic dimensions, NCHW
            ph = tf.placeholder(
                dtype=tf.float32, shape=(None, None, 5, None, None))
            assert_allclose(f(x, channels_last=False, keepdims=True, ph=ph), a1)
