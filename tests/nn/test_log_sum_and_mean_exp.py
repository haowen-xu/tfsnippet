import numpy as np
import tensorflow as tf

from tfsnippet.nn import *


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
