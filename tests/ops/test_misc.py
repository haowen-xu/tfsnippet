import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.ops import *


class AddNBroadcastTestCase(tf.test.TestCase):

    def test_add_n_broadcast(self):
        # test zero input
        with pytest.raises(ValueError, match='`tensors` must not be empty'):
            _ = add_n_broadcast([])

        with self.test_session() as sess:
            a = np.array([[1., 2.], [3., 4.]])
            b = np.array([5., 6.])
            c = np.array([[7., 8.]])
            a_tensor, b_tensor, c_tensor = \
                tf.constant(a), tf.constant(b), tf.constant(c)

            # test one input
            np.testing.assert_allclose(
                sess.run(add_n_broadcast([a_tensor])), a)

            # test two inputs
            np.testing.assert_allclose(
                sess.run(add_n_broadcast([a_tensor, b_tensor])), a + b)

            # test three inputs
            np.testing.assert_allclose(
                sess.run(add_n_broadcast([a_tensor, b_tensor, c_tensor])),
                a + b + c
            )


class LogSumAndMeanExpTestCase(tf.test.TestCase):

    def test_log_sum_exp(self):
        with self.test_session() as sess:
            x = np.linspace(0, 10, 1000).reshape([5, 10, 20])
            np.testing.assert_allclose(
                np.log(np.sum(np.exp(x), axis=-1, keepdims=False)),
                sess.run(log_sum_exp(x, axis=-1))
            )
            np.testing.assert_allclose(
                np.log(np.sum(np.exp(x), axis=(0, 2), keepdims=False)),
                sess.run(log_sum_exp(x, axis=(0, 2), keepdims=False))
            )
            np.testing.assert_allclose(
                np.log(np.sum(np.exp(x), axis=(0, 2), keepdims=True)),
                sess.run(log_sum_exp(x, axis=(0, 2), keepdims=True))
            )
            np.testing.assert_allclose(
                np.log(np.sum(np.exp(x), axis=None, keepdims=False)),
                sess.run(log_sum_exp(x, keepdims=False))
            )
            np.testing.assert_allclose(
                np.log(np.sum(np.exp(x), axis=None, keepdims=True)),
                sess.run(log_sum_exp(x, keepdims=True))
            )

    def test_log_mean_exp(self):
        with self.test_session() as sess:
            x = np.linspace(0, 10, 1000).reshape([5, 10, 20])
            np.testing.assert_allclose(
                np.log(np.mean(np.exp(x), axis=-1, keepdims=False)),
                sess.run(log_mean_exp(x, axis=-1))
            )
            np.testing.assert_allclose(
                np.log(np.mean(np.exp(x), axis=(0, 2), keepdims=False)),
                sess.run(log_mean_exp(x, axis=(0, 2), keepdims=False))
            )
            np.testing.assert_allclose(
                np.log(np.mean(np.exp(x), axis=(0, 2), keepdims=True)),
                sess.run(log_mean_exp(x, axis=(0, 2), keepdims=True))
            )
            np.testing.assert_allclose(
                np.log(np.mean(np.exp(x), axis=None, keepdims=False)),
                sess.run(log_mean_exp(x, keepdims=False))
            )
            np.testing.assert_allclose(
                np.log(np.mean(np.exp(x), axis=None, keepdims=True)),
                sess.run(log_mean_exp(x, keepdims=True))
            )
