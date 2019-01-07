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


class ImportanceSamplingLogLikelihoodTestCase(tf.test.TestCase):

    def test_error(self):
        with pytest.raises(ValueError,
                           match='importance sampling log-likelihood requires '
                                 'multi-samples of latent variables'):
            log_p, log_q = prepare_test_payload()
            _ = importance_sampling_log_likelihood(log_p, log_q, axis=None)

    def test_monto_carlo_objective(self):
        with self.test_session() as sess:
            log_p, log_q = prepare_test_payload()

            ll = importance_sampling_log_likelihood(log_p, log_q, axis=0)
            ll_shape = ll.get_shape().as_list()
            assert_allclose(*sess.run([
                ll,
                log_mean_exp(log_p - log_q, axis=0)
            ]))

            ll_k = importance_sampling_log_likelihood(
                log_p, log_q, axis=0, keepdims=True)
            self.assertListEqual(
                [1] + ll_shape, ll_k.get_shape().as_list())
            assert_allclose(*sess.run([
                ll_k,
                log_mean_exp(log_p - log_q, axis=0, keepdims=True)
            ]))
