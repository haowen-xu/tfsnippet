import pytest
import numpy as np
import tensorflow as tf

from tests.helper import assert_variables
from tfsnippet.layers import *
from tfsnippet.utils import get_static_shape


class WeightNormTestCase(tf.test.TestCase):

    def test_errors(self):
        kernel = tf.reshape(tf.range(6, dtype=tf.float32), [3, 2])
        scale = tf.zeros_like(kernel)

        with pytest.raises(ValueError, match='`use_scale` is False but '
                                             '`scale` is specified'):
            _ = weight_norm(kernel, -1, use_scale=False, scale=scale)

        with pytest.raises(ValueError, match='`axis` cannot be empty'):
            _ = weight_norm(kernel, ())

    def test_weight_norm(self):
        kernel = np.reshape(np.arange(120, dtype=np.float64), (2, 3, 4, 5))
        scale = np.random.normal(size=kernel.shape)

        with self.test_session() as sess:
            # test one dimension, without scale
            np.testing.assert_allclose(
                sess.run(weight_norm(tf.constant(kernel), axis=-1,
                                     use_scale=False)),
                kernel / np.sqrt(
                    np.sum(kernel ** 2, axis=(0, 1, 2), keepdims=True)),
                rtol=1e-5
            )

            # test two dimensions, without scale
            np.testing.assert_allclose(
                sess.run(weight_norm(tf.constant(kernel), axis=(0, -1),
                                     use_scale=False)),
                kernel / np.sqrt(
                    np.sum(kernel ** 2, axis=(1, 2), keepdims=True)),
                rtol=1e-5
            )

            # test one dimension, with given scale
            np.testing.assert_allclose(
                sess.run(weight_norm(tf.constant(kernel), axis=-1,
                                     scale=tf.constant(scale))),
                scale * kernel / np.sqrt(
                    np.sum(kernel ** 2, axis=(0, 1, 2), keepdims=True)),
                rtol=1e-5
            )

            # test two dimensions, with given scale
            np.testing.assert_allclose(
                sess.run(weight_norm(tf.constant(kernel), axis=(0, -1),
                                     scale=tf.constant(scale))),
                scale * kernel / np.sqrt(
                    np.sum(kernel ** 2, axis=(1, 2), keepdims=True)),
                rtol=1e-5
            )

            # test one dimension, with generated scale
            weight = weight_norm(tf.constant(kernel), axis=-1)
            scale_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]
            self.assertEqual(get_static_shape(scale_var), kernel.shape)
            sess.run(scale_var.assign(scale + 1.))

            np.testing.assert_allclose(
                sess.run(weight),
                (scale + 1.) * kernel / np.sqrt(
                    np.sum(kernel ** 2, axis=(0, 1, 2), keepdims=True)),
                rtol=1e-5
            )

            # test two dimensions, with generated scale
            weight = weight_norm(tf.constant(kernel), axis=(0, -1))
            scale_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[1]
            self.assertEqual(get_static_shape(scale_var), kernel.shape)
            sess.run(scale_var.assign(scale + 2.))

            np.testing.assert_allclose(
                sess.run(weight),
                (scale + 2.) * kernel / np.sqrt(
                    np.sum(kernel ** 2, axis=(1, 2), keepdims=True)),
                rtol=1e-5
            )

    def test_weight_norm_vars(self):
        # test trainable
        with tf.Graph().as_default():
            _ = weight_norm(tf.zeros([2, 3]), -1)
            assert_variables(['scale'], trainable=True, scope='weight_norm',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test non-trainable
        with tf.Graph().as_default():
            _ = weight_norm(tf.zeros([2, 3]), -1, trainable=False)
            assert_variables(['scale'], trainable=False, scope='weight_norm',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test no scale
        with tf.Graph().as_default():
            _ = weight_norm(tf.zeros([2, 3]), -1, use_scale=False)
            assert_variables(['scale'], exist=False, scope='weight_norm')
