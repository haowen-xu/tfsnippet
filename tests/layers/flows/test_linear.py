import functools

import numpy as np
import tensorflow as tf
from mock import mock

from tests.helper import assert_variables
from tests.layers.convolutional.test_conv2d import patched_conv2d
from tests.layers.flows.helper import invertible_flow_standard_check
from tfsnippet.layers import InvertibleDense, InvertibleConv2d


def naive_invertible_linear(input, kernel, axis, value_ndims):
    assert(len(kernel.shape) == 2)
    assert(kernel.shape[0] == kernel.shape[1])
    assert(value_ndims > 0)

    # transpose the shape of input if `axis` != -1
    input_shape = input.shape
    transpose_shape = list(range(len(input_shape)))
    transpose_shape[axis], transpose_shape[-1] = \
        transpose_shape[-1], transpose_shape[axis]

    # do linear mapping
    output = np.transpose(
        np.dot(np.transpose(input, transpose_shape), kernel),
        transpose_shape
    )
    assert(output.shape == input.shape)

    # compute the log-det
    log_det = np.linalg.slogdet(kernel)[1]
    log_det = log_det * (np.prod(input.shape[-value_ndims:]) /
                         input.shape[axis])
    log_det_shape = input.shape[:-value_ndims]
    log_det *= np.ones(log_det_shape, dtype=input.dtype)

    return output, log_det


class InvertibleDenseTestCase(tf.test.TestCase):

    def test_invertible_dense(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5)
        np.random.seed(1234)

        with self.test_session() as sess:
            x = np.random.normal(size=[3, 5, 7]).astype(np.float32)
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, 7])
            kernel = np.random.normal(size=(7, 7)).astype(x.dtype)
            y, log_det = naive_invertible_linear(x, kernel, -1, 1)

            layer = InvertibleDense(strict_invertible=False)
            y_out, log_det_out = layer.transform(x_ph)
            # The kernel is initialized as an orthogonal matrix.  As a result,
            # the log-det is zero (the computation result should be close to
            # zero, but may not be zero).  This is not good for testing.
            # Thus we initialize it with an arbitrary matrix.
            sess.run(tf.assign(layer._kernel_matrix._matrix, kernel))
            y_out, log_det_out = sess.run([y_out, log_det_out],
                                          feed_dict={x_ph: x})
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x}, rtol=1e-5,
                atol=1e-6
            )

            assert_variables(['matrix'], trainable=True,
                             scope='invertible_dense/kernel',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test non-trainable
        with tf.Graph().as_default():
            layer = InvertibleDense(strict_invertible=False, trainable=False)
            layer.apply(tf.zeros([2, 3, 4, 5]))
            assert_variables(['matrix'], trainable=False,
                             scope='invertible_dense/kernel',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])


class InvertibleConv2dTestCase(tf.test.TestCase):

    def test_invertible_conv2d(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)
        np.random.seed(1234)

        with self.test_session() as sess, \
                mock.patch('tensorflow.nn.conv2d', patched_conv2d):
            # test channels_last = True, static input
            x = np.random.normal(size=[3, 4, 5, 6, 7]).astype(np.float32)
            kernel = np.random.normal(size=(7, 7)).astype(x.dtype)
            y, log_det = naive_invertible_linear(x, kernel, -1, 3)

            layer = InvertibleConv2d(channels_last=True,
                                     strict_invertible=False)
            y_out, log_det_out = layer.transform(x)
            # The kernel is initialized as an orthogonal matrix.  As a result,
            # the log-det is zero (the computation result should be close to
            # zero, but may not be zero).  This is not good for testing.
            # Thus we initialize it with an arbitrary matrix.
            sess.run(tf.assign(layer._kernel_matrix._matrix, kernel))
            y_out, log_det_out = sess.run([y_out, log_det_out])
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(
                self, layer, sess, x, rtol=1e-4, atol=1e-5)

            assert_variables(['matrix'], trainable=True,
                             scope='invertible_conv2d/kernel',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

            # test channels_last = False, dynamic input
            x = np.random.normal(size=[3, 4, 5, 6, 7]).astype(np.float32)
            x_ph = tf.placeholder(dtype=tf.float32,
                                  shape=[None, None, 5, None, None])
            kernel = np.random.normal(size=(5, 5)).astype(x.dtype)
            y, log_det = naive_invertible_linear(x, kernel, -3, 3)

            layer = InvertibleConv2d(channels_last=False,
                                     strict_invertible=False)
            y_out, log_det_out = layer.transform(x_ph)
            # The kernel is initialized as an orthogonal matrix.  As a result,
            # the log-det is zero (the computation result should be close to
            # zero, but may not be zero).  This is not good for testing.
            # Thus we initialize it with an arbitrary matrix.
            sess.run(tf.assign(layer._kernel_matrix._matrix, kernel))
            y_out, log_det_out = sess.run([y_out, log_det_out],
                                          feed_dict={x_ph: x})
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x}, rtol=1e-4,
                atol=1e-5
            )

            assert_variables(['matrix'], trainable=True,
                             scope='invertible_conv2d/kernel',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test non-trainable
        with tf.Graph().as_default():
            layer = InvertibleConv2d(strict_invertible=False, trainable=False)
            layer.apply(tf.zeros([3, 4, 5, 6, 7]))
            assert_variables(['matrix'], trainable=False,
                             scope='invertible_conv2d/kernel',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])
