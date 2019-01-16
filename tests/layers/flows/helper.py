import numpy as np
import tensorflow as tf

from tfsnippet.layers import BaseFlow


class Ops(object):
    pass


tfops = Ops()
for attr in ('log', 'abs', 'exp', 'sign'):
    setattr(tfops, attr, getattr(tf, attr))

npyops = Ops()
for attr in ('log', 'abs', 'exp', 'sign'):
    setattr(npyops, attr, getattr(np, attr))


def safe_pow(ops, x, e):
    return ops.sign(x) * ops.exp(e * ops.log(ops.abs(x)))


def quadratic_transform(ops, x, a, b):
    return a * x ** 3 + b, ops.log(3. * a * (x ** 2))


def quadratic_inverse_transform(ops, y, a, b):
    return (
        safe_pow(ops, (y - b) / a, 1./3),
        ops.log(ops.abs(safe_pow(ops, (y - b) / a, -2. / 3) / (3. * a)))
    )


class QuadraticFlow(BaseFlow):

    def __init__(self, a, b, value_ndims=0):
        super(QuadraticFlow, self).__init__(x_value_ndims=value_ndims,
                                            y_value_ndims=value_ndims)
        self.a = a
        self.b = b

    def _build(self, input=None):
        pass

    @property
    def explicitly_invertible(self):
        return True

    def _transform(self, x, compute_y, compute_log_det):
        y, log_det = quadratic_transform(tfops, x, self.a, self.b)
        if self.x_value_ndims > 0:
            log_det = tf.reduce_sum(
                log_det, axis=tf.range(-self.x_value_ndims, 0, dtype=tf.int32))
        if not compute_y:
            y = None
        if not compute_log_det:
            log_det = None
        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        x, log_det = quadratic_inverse_transform(tfops, y, self.a, self.b)
        if self.y_value_ndims > 0:
            log_det = tf.reduce_sum(
                log_det, axis=tf.range(-self.y_value_ndims, 0, dtype=tf.int32))
        if not compute_x:
            x = None
        if not compute_log_det:
            log_det = None
        return x, log_det


def invertible_flow_standard_check(self, flow, session, x, feed_dict=None,
                                   atol=0., rtol=1e-5):
    x = tf.convert_to_tensor(x)
    self.assertTrue(flow.explicitly_invertible)

    # test mapping from x -> y -> x
    y, log_det_y = flow.transform(x)
    x2, log_det_x = flow.inverse_transform(y)

    x_out, y_out, log_det_y_out, x2_out, log_det_x_out = \
        session.run([x, y, log_det_y, x2, log_det_x], feed_dict=feed_dict)
    np.testing.assert_allclose(x2_out, x_out, atol=atol, rtol=rtol)

    np.testing.assert_allclose(
        -log_det_x_out, log_det_y_out, atol=atol, rtol=rtol)
    self.assertEqual(np.size(x_out), np.size(y_out))

    x_batch_shape = x_out.shape
    y_batch_shape = y_out.shape
    if flow.x_value_ndims > 0:
        x_batch_shape = x_batch_shape[:-flow.x_value_ndims]
    if flow.y_value_ndims > 0:
        y_batch_shape = y_batch_shape[:-flow.y_value_ndims]
    self.assertTupleEqual(log_det_y_out.shape, x_batch_shape)
    self.assertTupleEqual(log_det_x_out.shape, y_batch_shape)
    self.assertTupleEqual(log_det_y_out.shape, log_det_x_out.shape)
