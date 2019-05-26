import functools

import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.utils import get_static_shape, ensure_variables_initialized
from tfsnippet.variational import *
from tfsnippet.variational.estimators import (_vimco_replace_diag,
                                              _vimco_control_variate)


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


def log_mean_exp(x, axis, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max_reduced = x_max if keepdims else np.squeeze(x_max, axis=axis)
    out = x_max_reduced + np.log(
        np.mean(np.exp(x - x_max), axis=axis, keepdims=keepdims))
    return out


def slice_at(arr, axis, start, stop=None, step=None):
    if axis < 0:
        axis += len(arr.shape)
    s = (slice(None, None, None),) * axis + (slice(start, stop, step),)
    return arr[s]


def vimco_control_variate(log_f, axis):
    K = log_f.shape[axis]
    mean_except_k = (np.sum(log_f, axis=axis, keepdims=True) - log_f) / (K - 1)

    def sub_k(k):
        tmp = np.concatenate(
            [slice_at(log_f, axis, 0, k),
             slice_at(mean_except_k, axis, k, k + 1),
             slice_at(log_f, axis, k+1)],
            axis=axis
        )
        return log_mean_exp(tmp, axis=axis, keepdims=True)

    return np.concatenate([sub_k(k) for k in range(K)], axis=axis)


class VIMCOEstimatorTestCase(tf.test.TestCase):

    def test_vimco_replace_diag(self):
        with self.test_session() as sess:
            # 2-d
            x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            y = tf.constant([[10], [11], [12]])
            z = sess.run(_vimco_replace_diag(x, y, -2))
            np.testing.assert_equal(z, [[10, 2, 3], [4, 11, 6], [7, 8, 12]])

            # 4-d
            x = np.arange(4 * 3 * 3 * 5, dtype=np.int32).reshape([4, 3, 3, 5])
            y = -np.arange(4 * 3 * 1 * 5, dtype=np.int32).reshape([4, 3, 1, 5])
            x_ph = tf.placeholder(tf.int32, [None] * 4)
            y_ph = tf.placeholder(tf.int32, [None, None, 1, None])
            diag_mask = np.eye(3, 3).reshape([1, 3, 3, 1])
            z = sess.run(_vimco_replace_diag(
                tf.convert_to_tensor(x_ph), tf.convert_to_tensor(y_ph), -3),
                feed_dict={x_ph: x, y_ph: y}
            )
            np.testing.assert_equal(z, x * (1 - diag_mask) + y * diag_mask)

    def test_vimco_control_variate(self):
        with self.test_session() as sess:
            np.random.seed(1234)
            log_f = np.random.randn(4, 5, 6, 7).astype(np.float64)
            log_f_ph = tf.placeholder(tf.float64, [None] * 4)
            rank = len(log_f.shape)

            for axis in range(rank):
                out = sess.run(_vimco_control_variate(log_f, axis=axis - rank))
                out2 = sess.run(
                    _vimco_control_variate(log_f_ph, axis=axis - rank),
                    feed_dict={log_f_ph: log_f}
                )
                ans = vimco_control_variate(log_f, axis=axis - rank)
                np.testing.assert_allclose(out, ans)
                np.testing.assert_allclose(out2, ans)

    def test_error(self):
        x, y, z, f, log_f, log_q = \
            prepare_test_payload(is_reparameterized=False)

        with pytest.raises(ValueError,
                           match='vimco_estimator requires multi-samples of '
                                 'latent variables'):
            _ = vimco_estimator(log_f, log_q, axis=None)

        with pytest.raises(TypeError,
                           match=r'vimco_estimator only supports integer '
                                 r'`axis`: got \[0, 1\]'):
            _ = vimco_estimator(log_f, log_q, axis=[0, 1])

        with pytest.raises(ValueError,
                           match='`axis` out of range: rank 2 vs axis 2'):
            _ = vimco_estimator(log_f, log_q, axis=2)

        with pytest.raises(ValueError,
                           match='`axis` out of range: rank 2 vs axis -3'):
            _ = vimco_estimator(log_f, log_q, axis=-3)

        with pytest.raises(ValueError,
                           match='vimco_estimator only supports `log_values` '
                                 'with deterministic ndims'):
            _ = vimco_estimator(
                tf.placeholder(tf.float32, None),
                tf.zeros([1, 2]),
                axis=0
            )

        with pytest.raises(ValueError,
                           match='VIMCO requires sample size >= 2: '
                                 'sample axis is 0'):
            _ = vimco_estimator(
                tf.placeholder(tf.float32, [1, None]),
                tf.zeros([1, 2]),
                axis=0
            )

        with pytest.raises(Exception,
                           match='VIMCO requires sample size >= 2: '
                                 'sample axis is 1'):
            ph = tf.placeholder(tf.float32, [3, None])
            with tf.Session() as sess:
                sess.run(vimco_estimator(ph, tf.zeros([3, 1]), axis=1),
                         feed_dict={ph: np.zeros([3, 1])})

    def test_vimco(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        with self.test_session() as sess:
            x, y, z, f, log_f, log_q = \
                prepare_test_payload(is_reparameterized=False)

            # compute the gradient
            x_out, y_out, z_out, f_out, log_f_out, log_q_out = \
                sess.run([x, y, z, f, log_f, log_q])
            log_q_grad_out = (x_out ** 2 - 1) * 3 * (y_out ** 2)
            log_f_out = y_out * z_out

            t = np.sum(
                log_q_grad_out * (
                    log_mean_exp(log_f_out, axis=0, keepdims=True) -
                    vimco_control_variate(log_f_out, axis=0)
                ),
                axis=0
            )
            w_k_hat = f_out / np.sum(f_out, axis=0, keepdims=True)
            log_f_grad_out = z_out
            t += np.sum(
                w_k_hat * log_f_grad_out,
                axis=0
            )

            cost = vimco_estimator(log_f, log_q, axis=0)
            cost_shape = cost.get_shape().as_list()
            assert_allclose(sess.run(tf.gradients([cost], [y])[0]), t)

            cost_k = vimco_estimator(log_f, log_q, axis=0, keepdims=True)
            self.assertListEqual(
                [1] + cost_shape, cost_k.get_shape().as_list())
            assert_allclose(sess.run(tf.gradients([cost], [y])[0]), t)
