import numpy as np
import tensorflow as tf
from mock import mock

from tests.layers.convolutional.helper import (input_maybe_to_channels_last,
                                               output_maybe_to_channels_first)
from tfsnippet.ops import space_to_depth, depth_to_space

tf_space_to_depth = tf.space_to_depth
tf_depth_to_space = tf.depth_to_space


def patched_space_to_depth(input, block_size, data_format):
    input = input_maybe_to_channels_last(input, data_format=data_format)
    output = tf_space_to_depth(
        input=input, block_size=block_size, data_format='NHWC')
    output = output_maybe_to_channels_first(output, data_format=data_format)
    return output


def naive_space_to_depth(x, bs, channels_last):
    if channels_last:
        h, w, c = x.shape[-3:]
        h2, w2, c2 = h // bs, w // bs, c * bs * bs
        y = np.reshape(x, x.shape[:-3] + (h2, bs) + (w2, bs) + (c,))
        y = np.transpose(
            y, list(range(len(y.shape) - 5)) + [-5, -3, -4, -2, -1])
        y = np.reshape(y, (x.shape[:-3] + (h2, w2, c2)))
    else:
        c, h, w = x.shape[-3:]
        h2, w2, c2 = h // bs, w // bs, c * bs * bs
        y = np.reshape(x, x.shape[:-3] + (c,) + (h2, bs) + (w2, bs))
        y = np.transpose(
            y, list(range(len(y.shape) - 5)) + [-3, -1, -5, -4, -2])
        y = np.reshape(y, (x.shape[:-3] + (c2, h2, w2)))
    return y


def patched_depth_to_space(input, block_size, data_format):
    input = input_maybe_to_channels_last(input, data_format=data_format)
    output = tf_depth_to_space(
        input=input, block_size=block_size, data_format='NHWC')
    output = output_maybe_to_channels_first(output, data_format=data_format)
    return output


def naive_depth_to_space(x, bs, channels_last):
    if channels_last:
        h, w, c = x.shape[-3:]
        h2, w2, c2 = h * bs, w * bs, c // (bs * bs)
        y = np.reshape(x, x.shape[:-3] + (h, w, bs, bs, c2))
        y = np.transpose(
            y, list(range(len(y.shape) - 5)) + [-5, -3, -4, -2, -1])
        y = np.reshape(y, (x.shape[:-3] + (h2, w2, c2)))
    else:
        c, h, w = x.shape[-3:]
        h2, w2, c2 = h * bs, w * bs, c // (bs * bs)
        y = np.reshape(x, x.shape[:-3] + (bs, bs, c2, h, w))
        y = np.transpose(
            y, list(range(len(y.shape) - 5)) + [-3, -2, -5, -1, -4])
        y = np.reshape(y, (x.shape[:-3] + (c2, h2, w2)))
    return y


class SpaceToDepthTestCase(tf.test.TestCase):

    def test_space_to_depth(self):
        with self.test_session() as sess, \
                mock.patch('tensorflow.space_to_depth', patched_space_to_depth):
            # static shape, bs = 2, channels_last = True
            x = np.random.normal(size=[5, 8, 12, 7]).astype(np.float32)
            np.testing.assert_allclose(
                sess.run(space_to_depth(x, 2, True)),
                naive_space_to_depth(x, 2, True)
            )

            # static shape, bs = 3, channels_last = False
            x = np.random.normal(size=[4, 5, 7, 6, 9]).astype(np.float32)
            np.testing.assert_allclose(
                sess.run(space_to_depth(x, 3, False)),
                naive_space_to_depth(x, 3, False)
            )

            # dynamic shape, bs = 2, channels_last = True
            x = np.random.normal(size=[4, 5, 8, 12, 7]).astype(np.float32)
            x_ph = tf.placeholder(shape=[None, None, None, None, 7],
                                  dtype=tf.float32)
            np.testing.assert_allclose(
                sess.run(space_to_depth(x_ph, 2, True), feed_dict={x_ph: x}),
                naive_space_to_depth(x, 2, True)
            )

            # dynamic shape, bs = 3, channels_last = False
            x = np.random.normal(size=[5, 7, 6, 9]).astype(np.float32)
            x_ph = tf.placeholder(shape=[None, 7, None, None],
                                  dtype=tf.float32)
            np.testing.assert_allclose(
                sess.run(space_to_depth(x_ph, 3, False), feed_dict={x_ph: x}),
                naive_space_to_depth(x, 3, False)
            )


class DepthToSpaceTestCase(tf.test.TestCase):

    def test_depth_to_space(self):
        with self.test_session() as sess, \
                mock.patch('tensorflow.depth_to_space', patched_depth_to_space):
            # static shape, bs = 2, channels_last = True
            x = np.random.normal(size=[5, 4, 6, 28]).astype(np.float32)
            np.testing.assert_allclose(
                sess.run(depth_to_space(x, 2, True)),
                naive_depth_to_space(x, 2, True)
            )

            # static shape, bs = 3, channels_last = False
            x = np.random.normal(size=[4, 5, 63, 2, 3]).astype(np.float32)
            np.testing.assert_allclose(
                sess.run(depth_to_space(x, 3, False)),
                naive_depth_to_space(x, 3, False)
            )

            # dynamic shape, bs = 2, channels_last = True
            x = np.random.normal(size=[4, 5, 4, 6, 28]).astype(np.float32)
            x_ph = tf.placeholder(shape=[None, None, None, None, 28],
                                  dtype=tf.float32)
            np.testing.assert_allclose(
                sess.run(depth_to_space(x_ph, 2, True), feed_dict={x_ph: x}),
                naive_depth_to_space(x, 2, True)
            )

            # dynamic shape, bs = 3, channels_last = False
            x = np.random.normal(size=[5, 63, 2, 3]).astype(np.float32)
            x_ph = tf.placeholder(shape=[None, 63, None, None],
                                  dtype=tf.float32)
            np.testing.assert_allclose(
                sess.run(depth_to_space(x_ph, 3, False), feed_dict={x_ph: x}),
                naive_depth_to_space(x, 3, False)
            )
