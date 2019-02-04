import functools

import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.utils import get_static_shape, ensure_variables_initialized
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
    log_q = (x ** 2 - 1) * (y ** 3)
    return x, y, z, f, log_f, log_q


class SGVBEstimatorTestCase(tf.test.TestCase):

    def test_sgvb(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        with self.test_session() as sess:
            x, y, z, f, log_f, log_q = \
                prepare_test_payload(is_reparameterized=True)

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
            x, y, z, f, log_f, log_q = \
                prepare_test_payload(is_reparameterized=True)
            _ = iwae_estimator(log_f, axis=None)

    def test_iwae(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        with self.test_session() as sess:
            x, y, z, f, log_f, log_q = \
                prepare_test_payload(is_reparameterized=True)
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


class NVILEstimatorTestCase(tf.test.TestCase):

    def test_error(self):
        x, y, z, f, log_f, log_q = \
            prepare_test_payload(is_reparameterized=False)
        with pytest.raises(ValueError,
                           match='`baseline` is not specified, thus '
                                 '`center_by_moving_average` must be False'):
            _ = nvil_estimator(log_f, log_q, center_by_moving_average=False)

        with pytest.raises(ValueError,
                           match='The shape of `values` after `batch_axis` '
                                 'having been reduced must be static'):
            _ = nvil_estimator(
                tf.placeholder(dtype=tf.float32, shape=[None, None]),
                log_q,
                batch_axis=-1
            )

    def test_nvil(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        with self.test_session() as sess:
            x, y, z, f, log_f, log_q = \
                prepare_test_payload(is_reparameterized=False)
            baseline = 3.14 * tf.cos(y)
            alt_f = tf.exp(2 * y * z)

            # baseline is None, center by moving average, no sampling
            cost, baseline_cost = nvil_estimator(
                values=f,
                latent_log_joint=log_q
            )
            self.assertIsNone(baseline_cost)
            var_count = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            moving_mean = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]
            self.assertEqual(moving_mean.name, 'nvil_estimator/moving_mean:0')
            self.assertEqual(get_static_shape(moving_mean), (1, 1))
            ensure_variables_initialized()
            sess.run(tf.assign(moving_mean, [[6.]]))

            cost_shape = cost.get_shape().as_list()
            moving_mean = 4.8 + .2 * tf.reduce_mean(f)
            assert_allclose(*sess.run([
                tf.gradients([cost], [y])[0],
                tf.reduce_sum(
                    z * f +
                    (f - moving_mean) * (3 * (x ** 2 - 1) * (y ** 2)),
                    axis=0)
            ]))

            # baseline is given, no center by moving average
            cost, baseline_cost = nvil_estimator(
                values=f,
                latent_log_joint=log_q,
                baseline=baseline,
                center_by_moving_average=False,
            )
            self.assertEqual(
                len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
                var_count
            )
            self.assertListEqual(cost.get_shape().as_list(), cost_shape)

            assert_allclose(*sess.run([
                tf.gradients([cost], [y])[0],
                tf.reduce_sum(
                    z * f +
                    (f - 3.14 * tf.cos(y)) * (3 * (x ** 2 - 1) * (y ** 2)),
                    axis=0)
            ]))
            assert_allclose(*sess.run([
                tf.gradients([baseline_cost], [y])[0],
                # -2 * (f(x,z) - C(x)) * C'(x)
                tf.reduce_sum(
                    -2 * (f - baseline) * (-3.14 * tf.sin(y)),
                    axis=0
                )
            ]))

            # baseline is given, no center by moving average, axis = [0]
            cost, baseline_cost = nvil_estimator(
                values=f,
                latent_log_joint=log_q,
                baseline=baseline,
                center_by_moving_average=False,
                axis=[0]
            )
            self.assertEqual(
                len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
                var_count
            )
            self.assertListEqual(cost.get_shape().as_list(), cost_shape[1:])

            assert_allclose(*sess.run([
                tf.gradients([cost], [y])[0],
                tf.reduce_sum(
                    z * f +
                    (f - 3.14 * tf.cos(y)) * (3 * (x ** 2 - 1) * (y ** 2)),
                    axis=0) / 7
            ]))
            assert_allclose(*sess.run([
                tf.gradients([baseline_cost], [y])[0],
                # -2 * (f(x,z) - C(x)) * C'(x)
                tf.reduce_sum(
                    -2 * (f - baseline) * (-3.14 * tf.sin(y)),
                    axis=0) / 7
            ]))
