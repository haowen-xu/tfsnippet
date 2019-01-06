import copy
from contextlib import contextmanager

import numpy as np
import pytest

import mock
import tensorflow as tf

from tests.layers.flows.helper import invertible_flow_standard_check
from tfsnippet.layers import ActNorm, act_norm, act_norm_conv2d


def naive_act_norm_initialize(x, axis):
    """Compute the act_norm initial `scale` and `bias` for `x`."""
    x = np.asarray(x)
    axis = list(sorted(set([a + len(x.shape) if a < 0 else a for a in axis])))
    min_axis = np.min(axis)
    reduce_axis = tuple(a for a in range(len(x.shape)) if a not in axis)
    var_shape = [x.shape[a] if a in axis else 1
                 for a in range(min_axis, len(x.shape))]
    mean = np.reshape(np.mean(x, axis=reduce_axis, keepdims=True), var_shape)
    bias = -mean
    scale = 1. / np.reshape(
        np.sqrt(np.mean((x - mean) ** 2, axis=reduce_axis, keepdims=True)),
        var_shape
    )
    return scale, bias


def naive_act_norm_transform(x, value_ndims, scale, bias):
    y = (x + bias) * scale
    log_det = np.log(np.abs(scale)) * np.ones_like(x)
    if value_ndims > 0:
        log_det = np.sum(log_det, axis=tuple(range(-value_ndims, 0)))
    return y, log_det


def assert_allclose(a, b, epsilon=5e-4):
    assert(a.shape == b.shape)
    assert(np.max(np.abs(a - b)) <= epsilon)


class ActNormClassTestCase(tf.test.TestCase):

    def test_error(self):
        with pytest.raises(ValueError,
                           match='Invalid value for argument `scale_type`'):
            _ = ActNorm(-1, scale_type='xyz')

        with pytest.raises(ValueError, match='`axis` must not be empty'):
            _ = ActNorm(())

        with pytest.raises(ValueError,
                           match='`ActNorm` requires `input` to build'):
            _ = ActNorm(-1).build()

        with pytest.raises(ValueError,
                           match='Initializing ActNorm requires multiple '
                                 '`x` samples, thus `x` must have at least '
                                 'one more dimension than `var_shape`'):
            act_norm = ActNorm([-3, -1], initializing=True)
            _ = act_norm.apply(tf.zeros([2, 3, 4]))

    def test_act_norm(self):
        np.random.seed(1234)

        x = np.random.normal(size=[3, 4, 5, 6, 7])
        x_ph = tf.placeholder(dtype=tf.float64, shape=[None, None, 5, None, 7])
        x2 = np.random.normal(size=[2, 3, 4, 5, 6, 7])
        x2_ph = tf.placeholder(dtype=tf.float64,
                               shape=[None, None, None, 5, None, 7])
        x3 = np.random.normal(size=[4, 5, 6, 7])
        x3_ph = tf.placeholder(dtype=tf.float64, shape=[None, 5, None, 7])

        with self.test_session() as sess:
            # -- static input shape, scale_type = 'scale', value_ndims = 0
            axis = [-1, -3]
            value_ndims = 0
            var_shape = (5, 1, 7)

            scale, bias = naive_act_norm_initialize(x, axis)
            self.assertEqual(scale.shape, var_shape)
            self.assertEqual(bias.shape, var_shape)

            # test initialize
            act_norm = ActNorm(axis=axis, value_ndims=value_ndims,
                               scale_type='scale', initializing=True)
            y_out, log_det_out = sess.run(
                act_norm.transform(tf.constant(x, dtype=tf.float64)))
            self.assertEqual(act_norm._bias.dtype.base_dtype, tf.float64)

            scale_out, bias_out = sess.run([act_norm._scale, act_norm._bias])
            assert_allclose(scale_out, scale)
            assert_allclose(bias_out, bias)

            # test the transform output from the initializing procedure
            y, log_det = naive_act_norm_transform(x, value_ndims, scale, bias)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(log_det.shape, x.shape)
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)

            # test use an initialized act_norm
            y2, log_det2 = naive_act_norm_transform(
                x2, value_ndims, scale, bias)
            self.assertEqual(y2.shape, x2.shape)
            self.assertEqual(log_det2.shape, x2.shape)
            y2_out, log_det2_out = sess.run(
                act_norm.transform(tf.constant(x2, dtype=tf.float64)))
            assert_allclose(y2_out, y2)
            assert_allclose(log_det2_out, log_det2)

            # -- dynamic input shape, scale_type = 'log_scale', value_ndims = 2
            value_ndims = 2

            # test initialize
            act_norm = ActNorm(axis=axis, value_ndims=value_ndims,
                               scale_type='log_scale', initializing=True)
            y_out, log_det_out = sess.run(
                act_norm.transform(x_ph), feed_dict={x_ph: x})
            self.assertEqual(act_norm._bias.dtype.base_dtype, tf.float64)

            scale_out, bias_out = sess.run(
                [tf.exp(act_norm._log_scale), act_norm._bias])
            assert_allclose(scale_out, scale)
            assert_allclose(bias_out, bias)

            # test the transform output from the initializing procedure
            y, log_det = naive_act_norm_transform(x, value_ndims, scale, bias)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(log_det.shape, x.shape[:-value_ndims])
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)

            # test use an initialized act_norm
            y2, log_det2 = naive_act_norm_transform(
                x2, value_ndims, scale, bias)
            self.assertEqual(y2.shape, x2.shape)
            self.assertEqual(log_det2.shape, x2.shape[:-value_ndims])
            y2_out, log_det2_out = sess.run(
                act_norm.transform(x2_ph), feed_dict={x2_ph: x2})
            assert_allclose(y2_out, y2)
            assert_allclose(log_det2_out, log_det2)

            # invertible flow standard checks
            invertible_flow_standard_check(
                self, act_norm, sess, x_ph, feed_dict={x_ph: x})

            # -- dynamic input shape, scale_type = 'scale', value_ndims = 4
            value_ndims = 4

            # test initialize
            act_norm = ActNorm(axis=axis, value_ndims=value_ndims,
                               scale_type='scale', initializing=True)
            y_out, log_det_out = sess.run(
                act_norm.transform(x_ph), feed_dict={x_ph: x})
            self.assertEqual(act_norm._bias.dtype.base_dtype, tf.float64)

            scale_out, bias_out = sess.run([act_norm._scale, act_norm._bias])
            assert_allclose(scale_out, scale)
            assert_allclose(bias_out, bias)

            # test the transform output from the initializing procedure
            y, log_det = naive_act_norm_transform(x, value_ndims, scale, bias)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(log_det.shape, x.shape[:-value_ndims])
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)

            # test use an initialized act_norm
            y3, log_det3 = naive_act_norm_transform(
                x3, value_ndims, scale, bias)
            self.assertEqual(y3.shape, x3.shape)
            self.assertEqual(log_det3.shape, x3.shape[:-value_ndims])
            y3_out, log_det3_out = sess.run(
                act_norm.transform(x3_ph), feed_dict={x3_ph: x3})
            assert_allclose(y3_out, y3)
            assert_allclose(log_det3_out, log_det3)

            # invertible flow standard checks
            invertible_flow_standard_check(self, act_norm, sess, x)


class _ActNorm(ActNorm):

    captured = None

    def __init__(self, *args, **kwargs):
        ActNorm.__init__(self, *args, **kwargs)
        self.captured.append([
            copy.copy(args),
            copy.copy(kwargs),
            self
        ])
        self.apply = mock.Mock(self.apply)

    @classmethod
    @contextmanager
    def patch(cls):
        from tfsnippet.layers.normalization import act_norm_
        old_ActNorm = act_norm_.ActNorm
        try:
            act_norm_.ActNorm = _ActNorm
            cls.captured = []
            yield cls.captured
        finally:
            act_norm_.ActNorm = old_ActNorm


class ActNormFuncTestCase(tf.test.TestCase):

    def test_act_norm(self):
        x = tf.zeros([2, 3, 4])
        axis = (-2, -1)

        with _ActNorm.patch() as captured:
            _ = act_norm(x, axis=axis, value_ndims=3, epsilon=.5)
            args, kwargs, o = captured[-1]
            self.assertTupleEqual(args, ())
            self.assertDictEqual(
                kwargs, {'axis': axis, 'value_ndims': 3, 'epsilon': .5})
            self.assertEqual(o.apply.call_args, ((x,), {}))

    def test_act_norm_conv2d(self):
        x = tf.zeros([2, 3, 4, 5])

        with _ActNorm.patch() as captured:
            # channels_last = True
            _ = act_norm_conv2d(x, epsilon=.5)
            args, kwargs, o = captured[-1]
            self.assertTupleEqual(args, ())
            self.assertDictEqual(
                kwargs, {'axis': -1, 'value_ndims': 3, 'epsilon': .5})
            self.assertEqual(o.apply.call_args, ((x,), {}))

            # channels_last = False
            _ = act_norm_conv2d(x, channels_last=False, epsilon=.5)
            args, kwargs, o = captured[-1]
            self.assertTupleEqual(args, ())
            self.assertDictEqual(
                kwargs, {'axis': -3, 'value_ndims': 3, 'epsilon': .5})
            self.assertEqual(o.apply.call_args, ((x,), {}))
