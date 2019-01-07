import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.ops import log_mean_exp
from tfsnippet.variational import *


def prepare_test_payload():
    np.random.seed(1234)
    log_p = tf.constant(np.random.normal(size=[13]), dtype=tf.float32)
    log_q = tf.constant(np.random.normal(size=[7, 13]), dtype=tf.float32)
    return log_p, log_q


def assert_allclose(a, b):
    np.testing.assert_allclose(a, b, atol=1e-4)


class ELBOObjectiveTestCase(tf.test.TestCase):

    def test_elbo(self):
        with self.test_session() as sess:
            log_p, log_q = prepare_test_payload()

            obj = elbo_objective(log_p, log_q)
            obj_shape = obj.get_shape().as_list()
            assert_allclose(*sess.run([
                obj,
                log_p - log_q
            ]))

            obj_r = elbo_objective(log_p, log_q, axis=0)
            self.assertListEqual(
                obj_shape[1:], obj_r.get_shape().as_list())
            assert_allclose(*sess.run([
                obj_r,
                tf.reduce_mean(log_p - log_q, axis=0)
            ]))

            obj_rk = elbo_objective(log_p, log_q, axis=0, keepdims=True)
            self.assertListEqual(
                [1] + obj_shape[1:], obj_rk.get_shape().as_list())
            assert_allclose(*sess.run([
                obj_rk,
                tf.reduce_mean(log_p - log_q, axis=0, keepdims=True)
            ]))


class MonteCarloObjectiveTestCase(tf.test.TestCase):

    def test_error(self):
        with pytest.raises(ValueError,
                           match='monte carlo objective requires multi-samples '
                                 'of latent variables'):
            log_p, log_q = prepare_test_payload()
            _ = monte_carlo_objective(log_p, log_q, axis=None)

    def test_monto_carlo_objective(self):
        with self.test_session() as sess:
            log_p, log_q = prepare_test_payload()

            obj = monte_carlo_objective(log_p, log_q, axis=0)
            obj_shape = obj.get_shape().as_list()
            assert_allclose(*sess.run([
                obj,
                log_mean_exp(log_p - log_q, axis=0)
            ]))

            obj_k = monte_carlo_objective(log_p, log_q, axis=0, keepdims=True)
            self.assertListEqual(
                [1] + obj_shape, obj_k.get_shape().as_list())
            assert_allclose(*sess.run([
                obj_k,
                log_mean_exp(log_p - log_q, axis=0, keepdims=True)
            ]))
