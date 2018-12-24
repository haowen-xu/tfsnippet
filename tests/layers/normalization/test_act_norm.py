import copy
from contextlib import contextmanager

import numpy as np
import pytest

import mock
import tensorflow as tf

from tests.layers.flows.helper import invertible_flow_standard_check
from tfsnippet.layers import ActNorm, act_norm, act_norm_conv2d
from tfsnippet.utils import global_reuse


def _compute(x, reduce_axis, var_shape, log_det_factor):
    mean = np.reshape(np.mean(x, axis=reduce_axis, keepdims=True), var_shape)
    bias = -mean
    scale = 1. / np.reshape(np.sqrt(np.mean((x - mean) ** 2, axis=reduce_axis)),
                            var_shape)
    y = (x + bias) * scale
    log_det = np.sum(np.log(np.abs(scale))) * log_det_factor
    if len(x.shape) > len(var_shape):
        log_det = np.reshape(
            log_det,
            [1] * (len(x.shape) - len(var_shape)) + list(log_det.shape)
        )
    return bias, scale, y, log_det


def _assert_allclose(*args, **kwargs):
    kwargs.setdefault('rtol', 1e-5)
    return np.testing.assert_allclose(*args, **kwargs)


class ActNormClassTestCase(tf.test.TestCase):

    def test_initializing(self):
        shape = (6, 4, 3, 2)
        var_shape = (4, 1, 2)
        reduce_axis = (0, 2)
        x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        bias, scale, y, _ = _compute(x, reduce_axis, var_shape, 3.)

        # test initialize scale & bias
        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True,
                scale_type='scale',
            )
            self.assertEqual(act_norm.value_ndims, 3)
            y_out = sess.run(act_norm(x))
            bias_out, scale_out = sess.run([act_norm._bias, act_norm._scale])

            self.assertEqual(y_out.shape, shape)
            self.assertEqual(bias_out.shape, var_shape)
            self.assertEqual(scale_out.shape, var_shape)

            _assert_allclose(y_out, y)
            _assert_allclose(bias_out, bias)
            _assert_allclose(scale_out, scale)

            # use an initialized act_norm will not initialize it again
            _ = sess.run(act_norm(np.random.uniform(size=shape)))
            bias_out, scale_out = sess.run([act_norm._bias, act_norm._scale])
            _assert_allclose(bias_out, bias)
            _assert_allclose(scale_out, scale)

        # test initialize log_scale & bias
        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True,
                scale_type='log_scale',
            )
            self.assertEqual(act_norm.value_ndims, 3)
            y_out = sess.run(act_norm(x))
            bias_out, log_scale_out = sess.run(
                [act_norm._bias, act_norm._log_scale])
            scale_out = np.exp(log_scale_out)

            self.assertEqual(y_out.shape, shape)
            self.assertEqual(bias_out.shape, var_shape)
            self.assertEqual(scale_out.shape, var_shape)

            _assert_allclose(y_out, y)
            _assert_allclose(bias_out, bias)
            _assert_allclose(scale_out, scale)

        # test initialization requires batch dimension
        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True
            )
            with pytest.raises(TypeError,
                               match='Initializing ActNorm requires multiple '
                                     '`x` samples, thus `x` must have at least '
                                     'one more dimension than `var_shape`'):
                _ = sess.run(act_norm(x[0]))

        # test require initializing on inverse transform
        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True
            )
            with pytest.raises(RuntimeError,
                               match='has not been initialized by data'):
                _ = sess.run(act_norm.inverse_transform(x[0]))

        # test initialization with extra sampling dimension
        shape = (3, 2, 4, 3, 2)
        x = np.reshape(x, shape)
        var_shape = (4, 1, 2)
        reduce_axis = (0, 1, 3)
        bias, scale, y, _ = _compute(x, reduce_axis, var_shape, 3.)

        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True,
                scale_type='scale',
            )
            self.assertEqual(act_norm.value_ndims, 3)
            y_out = sess.run(act_norm(x))
            bias_out, scale_out = sess.run([act_norm._bias, act_norm._scale])

            self.assertEqual(y_out.shape, shape)
            self.assertEqual(bias_out.shape, var_shape)
            self.assertEqual(scale_out.shape, var_shape)

            _assert_allclose(y_out, y)
            _assert_allclose(bias_out, bias)
            _assert_allclose(scale_out, scale)

    def test_use_initialized(self):
        @global_reuse
        def f(x, initializing=False):
            act_norm = ActNorm(var_shape=(3,), initializing=initializing,
                               scale_type='scale')
            self.assertEqual(act_norm.value_ndims, 1)
            return act_norm._bias, act_norm._scale, act_norm(x)

        with self.test_session() as sess:
            x = np.arange(18, dtype=np.float64).reshape((6, 3))
            x1, x2, x3 = x[:3, ...], x[3: 5, ...], x[5, ...]
            reduce_axis = (0,)
            var_shape = (3,)

            # compute bias & scale
            bias, scale, y, _ = _compute(x1, reduce_axis, var_shape, 3.)

            # initialize
            b_out, s_out, y_out = f(x1, initializing=True)
            y_out = sess.run(y_out)
            self.assertEqual(y_out.shape, (3, 3))
            b_out, s_out = sess.run([b_out, s_out])
            _assert_allclose(b_out, bias)
            _assert_allclose(s_out, scale)
            _assert_allclose(y_out, y)

            # compute y2 = f(x2), use parameters initialized by x1
            b_out, s_out, y_out = sess.run(f(x2, initializing=False))
            self.assertEqual(y_out.shape, (2, 3))
            _assert_allclose(b_out, bias)
            _assert_allclose(s_out, scale)
            _assert_allclose(y_out, (x2 + bias) * scale)

            # compute y3 = f(x3), use parameters initialized by x1
            b_out, s_out, y_out = sess.run(f(x3, initializing=False))
            self.assertEqual(y_out.shape, (3,))
            _assert_allclose(y_out, (x3 + bias) * scale)

    def test_flow(self):
        shape = (6, 4, 3, 2)
        var_shape = (4, 1, 2)
        reduce_axis = (0, 2)
        x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        _, _, y, log_det = _compute(x, reduce_axis, var_shape, 3.)

        # test initialized via log scale
        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True,
                scale_type='log_scale',
            )
            y_out, log_det_out = sess.run(act_norm.transform(x))
            self.assertEqual(y_out.shape, shape)
            self.assertEqual(log_det_out.shape, (1,))
            _assert_allclose(y_out, y)
            _assert_allclose(log_det_out, log_det)
            invertible_flow_standard_check(self, act_norm, sess, x)

        # test initialized via scale
        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True,
                scale_type='scale',
            )
            y_out, log_det_out = sess.run(act_norm.transform(x))
            self.assertEqual(y_out.shape, shape)
            self.assertEqual(log_det_out.shape, (1,))
            _assert_allclose(y_out, y)
            _assert_allclose(log_det_out, log_det)
            invertible_flow_standard_check(self, act_norm, sess, x)

        # test log-det of scalars
        var_shape = ()
        reduce_axis = (0, 1, 2, 3)
        x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        bias, scale, y, log_det = _compute(x, reduce_axis, var_shape, 1.)
        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True,
                scale_type='scale',
            )
            y_out, log_det_out = sess.run(act_norm.transform(x))
            self.assertEqual(y_out.shape, shape)
            self.assertEqual(log_det_out.shape, (1, 1, 1, 1))
            _assert_allclose(y_out, y)
            _assert_allclose(log_det_out, log_det)
            invertible_flow_standard_check(self, act_norm, sess, x)

            # additional check, because it's not checked before
            bias_out, scale_out = sess.run([act_norm._bias, act_norm._scale])
            self.assertEqual(bias_out.shape, var_shape)
            self.assertEqual(scale_out.shape, var_shape)
            _assert_allclose(bias_out, bias)
            _assert_allclose(scale_out, scale)

    def test_dynamic_x_shape(self):
        shape = (6, 4, 3, 2)
        var_shape = (4, 1, 2)
        reduce_axis = (0, 2)
        x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        x_in = tf.placeholder(dtype=tf.float32, shape=(None, 4, None, 2))
        bias, scale, y, log_det = _compute(x, reduce_axis, var_shape, 3.)

        with self.test_session() as sess:
            act_norm = ActNorm(
                var_shape=var_shape,
                initializing=True,
                scale_type='scale',
            )
            self.assertEqual(act_norm.value_ndims, 3)
            y_out, log_det_out = sess.run(
                act_norm.transform(x_in), feed_dict={x_in: x})
            bias_out, scale_out = sess.run([act_norm._bias, act_norm._scale])

            self.assertEqual(y_out.shape, shape)
            self.assertEqual(log_det_out.shape, (1,))
            self.assertEqual(bias_out.shape, var_shape)
            self.assertEqual(scale_out.shape, var_shape)

            _assert_allclose(y_out, y)
            _assert_allclose(log_det_out, log_det)
            _assert_allclose(bias_out, bias)
            _assert_allclose(scale_out, scale)

            invertible_flow_standard_check(
                self, act_norm, sess, x_in, feed_dict={x_in: x})


class _ActNorm(ActNorm):

    captured = None

    def __init__(self, *args, **kwargs):
        ActNorm.__init__(self, *args, **kwargs)
        self.captured.append([
            copy.copy(args),
            copy.copy(kwargs),
            self
        ])

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
        shape = (6, 4, 3, 2)
        x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

        # test len(var_shape) == 0
        with _ActNorm.patch() as captured:
            _ = act_norm(x, axis=(), value_ndims=0, epsilon=.5)
            args, kwargs, o = captured[0]
            self.assertTupleEqual(args, ())
            self.assertDictEqual(kwargs, {'var_shape': (), 'epsilon': .5})
            self.assertEqual(o._epsilon, .5)

        # test len(var_shape) == 1
        with _ActNorm.patch() as captured:
            _ = act_norm(x, axis=-1, value_ndims=1, dtype=tf.float64)
            args, kwargs, o = captured[0]
            self.assertTupleEqual(args, ())
            self.assertDictEqual(
                kwargs, {'var_shape': (2,), 'dtype': tf.float64})
            self.assertEqual(o.dtype, tf.float64)

        # test len(var_shape) > 1
        with _ActNorm.patch() as captured:
            _ = act_norm(x, axis=[-1, -3], value_ndims=3, trainable=False)
            args, kwargs, o = captured[0]
            self.assertTupleEqual(args, ())
            self.assertDictEqual(
                kwargs, {'var_shape': (4, 1, 2), 'trainable': False})
            self.assertNotIn(
                o._bias, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        # test unknown shape is invalid
        with pytest.raises(TypeError,
                           match='The ndims of `input` must be known'):
            _ = act_norm(tf.placeholder(shape=None, dtype=tf.float32))

        # test rank(x) >= value_ndims
        with pytest.raises(TypeError,
                           match='The `input` ndims must be larger than '
                                 '`value_ndims`'):
            _ = act_norm(x, value_ndims=5)

    def test_act_norm_conv2d(self):
        shape = (6, 4, 3, 2)
        x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

        with mock.patch('tfsnippet.layers.normalization.act_norm_.act_norm') \
                as m:
            _ = act_norm_conv2d(x, channels_last=False, trainable=False)
            self.assertTupleEqual(m.call_args[0], (x,))
            self.assertDictEqual(m.call_args[1], {
                'axis': -3, 'trainable': False, 'value_ndims': 3
            })

            _ = act_norm_conv2d(x, channels_last=True, trainable=False)
            self.assertTupleEqual(m.call_args[0], (x,))
            self.assertDictEqual(m.call_args[1], {
                'axis': -1, 'trainable': False, 'value_ndims': 3
            })
