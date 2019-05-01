import functools

import numpy as np
import pytest
import tensorflow as tf

from tests.layers.flows.helper import invertible_flow_standard_check
from tfsnippet.layers import CouplingLayer, conv2d
from tfsnippet.ops import (flatten_to_ndims,
                           unflatten_from_ndims,
                           transpose_conv2d_channels_last_to_x,
                           transpose_conv2d_channels_x_to_last)


def naive_coupling_layer(shift_and_scale_fn, x,
                         axis=-1, value_ndims=1, secondary=False,
                         scale_type='linear', sigmoid_scale_bias=2.,
                         reverse=False):
    assert(axis < 0)
    assert(axis >= -value_ndims)

    # split x into two halves
    n1 = x.shape[axis] // 2
    x1, x2 = np.split(x, [n1], axis)
    if secondary:
        x1, x2 = x2, x1
    n2 = x2.shape[axis]

    # compute the shift and scale
    shift, scale = shift_and_scale_fn(x1, n2)
    assert((scale_type is None) == (scale is None))

    # compose the output and log_det
    def safe_sigmoid(t):
        t = t + sigmoid_scale_bias

        # 1 / (1 + exp(-t)) = exp(-log(1 + exp(-t))
        ret = np.zeros_like(t)

        # for negative t, use `sigmoid(t) = exp(t) / (1 + exp(t))`
        neg_t_indices = t < 0
        exp_t = np.exp(t[neg_t_indices])
        ret[neg_t_indices] = exp_t / (1. + exp_t)

        # for positive t, use `sigmoid(t) = 1 / (1 + exp(-t))`
        pos_t_indices = t >= 0
        exp_neg_t = np.exp(-t[pos_t_indices])
        ret[pos_t_indices] = 1. / (1. + exp_neg_t)

        return ret

    y1 = x1
    if scale_type is not None:
        scale_functions = {
            'exp': np.exp,
            'sigmoid': safe_sigmoid,
            'linear': (lambda t: t),
        }
        scale = scale_functions[scale_type](scale)
        if reverse:
            y2 = x2 / scale - shift
            log_det = -np.log(np.abs(scale))
        else:
            y2 = (x2 + shift) * scale
            log_det = np.log(np.abs(scale))
    else:
        if reverse:
            y2 = x2 - shift
        else:
            y2 = x2 + shift
        log_det = np.zeros_like(x)

    if secondary:
        y1, y2 = y2, y1

    y = np.concatenate([y1, y2], axis=axis)
    log_det = np.sum(log_det, axis=tuple(range(-value_ndims, 0)))

    return y, log_det


class CouplingLayerTestCase(tf.test.TestCase):

    def test_coupling_layer(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5)

        np.random.seed(1234)
        kernel1 = np.random.normal(size=[3, 2]).astype(np.float32)
        kernel2 = np.random.normal(size=[2, 3]).astype(np.float32)
        shift1 = np.random.normal(size=[2]).astype(np.float32)
        shift2 = np.random.normal(size=[3]).astype(np.float32)

        def shift_and_scale_fn(x1, n2, no_scale=False):
            kernel = kernel1 if n2 == 2 else kernel2
            shift = tf.convert_to_tensor(shift1 if n2 == 2 else shift2)
            assert(kernel.shape[-1] == n2)
            assert(shift.shape[-1] == n2)
            x1, s1, s2 = flatten_to_ndims(x1, 2)
            scale = unflatten_from_ndims(tf.matmul(x1, kernel), s1, s2)
            shift = shift + tf.zeros_like(scale, dtype=shift.dtype)
            if no_scale:
                scale = None
            return shift, scale

        def shift_and_scale_numpy_fn(x1, n2, no_scale=False):
            a, b = shift_and_scale_fn(x1, n2, no_scale=no_scale)
            if b is None:
                a = sess.run(a)
            else:
                a, b = sess.run([a, b])
            return a, b

        with self.test_session() as sess:
            # test linear scale, primary
            x = np.random.normal(size=[3, 4, 5]).astype(np.float32)
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, 5])

            axis = -1
            value_ndims = 1
            y_ans, log_det_ans = naive_coupling_layer(
                shift_and_scale_numpy_fn, x, axis=axis,
                value_ndims=value_ndims, secondary=False, scale_type='linear',
                reverse=False
            )

            layer = CouplingLayer(
                shift_and_scale_fn, axis=axis, value_ndims=value_ndims,
                secondary=False, scale_type='linear'
            )
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            assert_allclose(y_out, y_ans)
            assert_allclose(log_det_out, log_det_ans)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x})

            # test exp scale, primary
            axis = -1
            value_ndims = 2
            y_ans, log_det_ans = naive_coupling_layer(
                shift_and_scale_numpy_fn, x, axis=axis,
                value_ndims=value_ndims, secondary=False, scale_type='exp',
                reverse=False
            )

            layer = CouplingLayer(
                shift_and_scale_fn, axis=axis, value_ndims=value_ndims,
                secondary=False, scale_type='exp'
            )
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            assert_allclose(y_out, y_ans)
            assert_allclose(log_det_out, log_det_ans)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x})

            # test sigmoid scale, secondary
            sigmoid_scale_bias = np.exp(1)

            axis = -1
            value_ndims = 1
            y_ans, log_det_ans = naive_coupling_layer(
                shift_and_scale_numpy_fn, x, axis=axis,
                value_ndims=value_ndims, secondary=False, scale_type='sigmoid',
                sigmoid_scale_bias=sigmoid_scale_bias, reverse=False
            )

            layer = CouplingLayer(
                shift_and_scale_fn, axis=axis, value_ndims=value_ndims,
                secondary=False, scale_type='sigmoid',
                sigmoid_scale_bias=sigmoid_scale_bias
            )
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            assert_allclose(y_out, y_ans)
            assert_allclose(log_det_out, log_det_ans)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x})

            # test None scale, primary
            axis = -1
            value_ndims = 1
            y_ans, log_det_ans = naive_coupling_layer(
                functools.partial(shift_and_scale_numpy_fn, no_scale=True),
                x, axis=axis,
                value_ndims=value_ndims, secondary=False, scale_type=None,
                reverse=False
            )

            layer = CouplingLayer(
                functools.partial(shift_and_scale_fn, no_scale=True),
                axis=axis, value_ndims=value_ndims,
                secondary=False, scale_type=None
            )
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            assert_allclose(y_out, y_ans)
            assert_allclose(log_det_out, log_det_ans)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x})

            # test None scale, secondary
            axis = -1
            value_ndims = 3
            y_ans, log_det_ans = naive_coupling_layer(
                functools.partial(shift_and_scale_numpy_fn, no_scale=True),
                x, axis=axis,
                value_ndims=value_ndims, secondary=True, scale_type=None,
                reverse=False
            )

            layer = CouplingLayer(
                functools.partial(shift_and_scale_fn, no_scale=True),
                axis=axis, value_ndims=value_ndims,
                secondary=True, scale_type=None
            )
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            assert_allclose(y_out, y_ans)
            assert_allclose(log_det_out, log_det_ans)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x})

    def test_coupling_layer_with_conv2d(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, atol=1e-5, rtol=5e-4)

        np.random.seed(1234)
        kernel1 = np.random.normal(size=[3, 3, 5, 6]).astype(np.float32)
        kernel2 = np.random.normal(size=[3, 3, 6, 5]).astype(np.float32)
        shift1 = np.random.normal(size=[6]).astype(np.float32)
        shift2 = np.random.normal(size=[5]).astype(np.float32)

        def shift_and_scale_fn(x1, n2, no_scale=False, channels_last=True):
            kernel = kernel1 if n2 == 6 else kernel2
            shift = tf.convert_to_tensor(shift1 if n2 == 6 else shift2)
            assert (kernel.shape[-1] == n2)
            assert (shift.shape[-1] == n2)

            x1 = transpose_conv2d_channels_x_to_last(
                x1, channels_last=channels_last
            )
            scale = conv2d(x1, n2, (3, 3), use_bias=False, kernel=kernel,
                           channels_last=True)
            shift = shift + tf.zeros_like(scale, dtype=shift.dtype)
            scale = transpose_conv2d_channels_last_to_x(scale, channels_last)
            shift = transpose_conv2d_channels_last_to_x(shift, channels_last)

            if no_scale:
                scale = None
            return shift, scale

        def shift_and_scale_numpy_fn(x1, n2, no_scale=False,
                                     channels_last=True):
            a, b = shift_and_scale_fn(x1, n2, no_scale, channels_last)
            if b is None:
                a = sess.run(a)
            else:
                a, b = sess.run([a, b])
            return a, b

        with self.test_session() as sess:
            # test exp scale, primary, NHWC
            x = np.random.normal(size=[11, 13, 32, 31, 11]).astype(np.float32)
            x_ph = tf.placeholder(dtype=tf.float32,
                                  shape=[None, None, None, None, 11])

            axis = -1
            value_ndims = 3
            y_ans, log_det_ans = naive_coupling_layer(
                shift_and_scale_numpy_fn, x, axis=axis,
                value_ndims=value_ndims, secondary=False, scale_type='exp',
                reverse=False
            )

            layer = CouplingLayer(
                shift_and_scale_fn, axis=axis, value_ndims=value_ndims,
                secondary=False, scale_type='exp'
            )
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            assert_allclose(y_out, y_ans)
            assert_allclose(log_det_out, log_det_ans)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x}, rtol=5e-4,
                atol=1e-5
            )

            # test sigmoid scale, secondary, NCHW
            x = np.transpose(x, [0, 1, 4, 2, 3])
            x_ph = tf.placeholder(dtype=tf.float32,
                                  shape=[None, None, 11, None, None])

            axis = -3
            value_ndims = 3
            y_ans, log_det_ans = naive_coupling_layer(
                functools.partial(shift_and_scale_numpy_fn,
                                  channels_last=False),
                x, axis=axis,
                value_ndims=value_ndims, secondary=True, scale_type='sigmoid',
                reverse=False
            )

            layer = CouplingLayer(
                functools.partial(shift_and_scale_fn, channels_last=False),
                axis=axis, value_ndims=value_ndims,
                secondary=True, scale_type='sigmoid'
            )
            y, log_det = layer.transform(x_ph)
            y_out, log_det_out = sess.run([y, log_det], feed_dict={x_ph: x})

            assert_allclose(y_out, y_ans)
            assert_allclose(log_det_out, log_det_ans)

            invertible_flow_standard_check(
                self, layer, sess, x_ph, feed_dict={x_ph: x}, rtol=5e-4,
                atol=1e-5
            )

    def test_errors(self):
        def shift_and_scale_fn(x1, n2):
            return tf.constant(0.), None

        with pytest.raises(ValueError, match='The feature axis of `input` must '
                                             'be at least 2'):
            layer = CouplingLayer(shift_and_scale_fn, axis=-1, value_ndims=1)
            _ = layer.apply(tf.zeros([2, 1]))

        with pytest.raises(RuntimeError, match='`scale_type` != None, but no '
                                               'scale is computed'):
            layer = CouplingLayer(shift_and_scale_fn, scale_type='linear')
            _ = layer.apply(tf.zeros([2, 3]))

        def shift_and_scale_fn(x1, n2):
            return tf.constant(0.), tf.constant(0.)

        with pytest.raises(RuntimeError, match='`scale_type` == None, but '
                                               'scale is computed'):
            layer = CouplingLayer(shift_and_scale_fn, scale_type=None)
            _ = layer.apply(tf.zeros([2, 3]))
