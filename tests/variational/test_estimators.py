import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.variational import *


def prepare_test_payload(is_reparameterized):
    np.random.seed(1234)
    x = tf.constant(np.random.normal(size=[7, 13]), dtype=tf.float32)  # input
    y = tf.constant(np.random.normal(size=[13]), dtype=tf.float32)  # param
    if is_reparameterized:
        z = y * x  # sample
    else:
        z = tf.stop_gradient(y) * x
    f = tf.exp(y * z)
    log_f = y * z
    return x, y, z, f, log_f


def assert_allclose(a, b):
    np.testing.assert_allclose(a, b, atol=1e-4)


class SGVBEstimatorTestCase(tf.test.TestCase):

    def test_sgvb(self):
        with self.test_session() as sess:
            x, y, z, f, log_f = prepare_test_payload(is_reparameterized=True)

            cost = sgvb_estimator(f)
            cost_shape = cost.get_shape().as_list()
            assert_allclose(*sess.run([
                tf.gradients([cost], [y])[0],
                tf.reduce_sum(2 * x * y * f, axis=0)
            ]))

            cost_r = sgvb_estimator(f, axis=0)
            self.assertListEqual(
                cost_shape[1:], cost_r.get_shape().as_list())
            assert_allclose(*sess.run([
                tf.gradients([cost_r], [y])[0],
                tf.reduce_sum(2 * x * y * f, axis=0) / 7
            ]))

            cost_rk = sgvb_estimator(f, axis=0, keepdims=True)
            self.assertListEqual(
                [1] + cost_shape[1:], cost_rk.get_shape().as_list())
            assert_allclose(*sess.run([
                tf.gradients([cost_rk], [y])[0],
                tf.reduce_sum(2 * x * y * f, axis=0) / 7
            ]))


class IWAEEstimatorTestCase(tf.test.TestCase):

    def test_error(self):
        with pytest.raises(ValueError,
                           match='iwae estimator requires multi-samples of '
                                 'latent variables'):
            x, y, z, f, log_f = prepare_test_payload(is_reparameterized=True)
            _ = iwae_estimator(log_f, axis=None)

    def test_iwae(self):
        with self.test_session() as sess:
            x, y, z, f, log_f = prepare_test_payload(is_reparameterized=True)
            wk_hat = f / tf.reduce_sum(f, axis=0, keepdims=True)

            cost = iwae_estimator(log_f, axis=0)
            cost_shape = cost.get_shape().as_list()
            assert_allclose(*sess.run([
                tf.gradients([cost], [y])[0],
                tf.reduce_sum(wk_hat * (2 * x * y), axis=0)
            ]))

            cost_k = iwae_estimator(log_f, axis=0, keepdims=True)
            self.assertListEqual(
                [1] + cost_shape, cost_k.get_shape().as_list())
            assert_allclose(*sess.run([
                tf.gradients([cost], [y])[0],
                tf.reduce_sum(wk_hat * (2 * x * y), axis=0)
            ]))
