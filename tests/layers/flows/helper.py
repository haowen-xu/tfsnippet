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

    def __init__(self, a, b):
        super(QuadraticFlow, self).__init__()
        self.a = a
        self.b = b

    def _build(self, input=None):
        pass

    @property
    def explicitly_invertible(self):
        return True

    def _transform(self, x, compute_y, compute_log_det, previous_log_det):
        y, log_det = quadratic_transform(tfops, x, self.a, self.b)
        if not compute_y:
            y = None
        if not compute_log_det:
            log_det = None
        elif previous_log_det is not None:
            log_det = previous_log_det + log_det
        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det,
                           previous_log_det):
        x, log_det = quadratic_inverse_transform(tfops, y, self.a, self.b)
        if not compute_x:
            x = None
        if not compute_log_det:
            log_det = None
        elif previous_log_det is not None:
            log_det = previous_log_det + log_det
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

    if flow.value_ndims > 0:
        log_det_shape = x_out.shape[:-flow.value_ndims]
    else:
        log_det_shape = x_out.shape
    self.assertTupleEqual(log_det_y_out.shape, log_det_shape)
    self.assertTupleEqual(log_det_x_out.shape, log_det_shape)

    # test with previous_log_det
    previous_log_det_y = 10. * np.random.normal(
        size=log_det_y_out.shape).astype(log_det_y_out.dtype)
    previous_log_det_x = 10. * np.random.normal(
        size=log_det_x_out.shape).astype(log_det_x_out.dtype)

    np.testing.assert_allclose(
        session.run(
            flow.transform(x, previous_log_det=previous_log_det_y)[1],
            feed_dict=feed_dict
        ),
        log_det_y_out + previous_log_det_y,
        atol=atol, rtol=rtol
    )

    np.testing.assert_allclose(
        session.run(
            flow.inverse_transform(y, previous_log_det=previous_log_det_x)[1],
            feed_dict=feed_dict
        ),
        log_det_x_out + previous_log_det_x,
        atol=atol, rtol=rtol
    )
