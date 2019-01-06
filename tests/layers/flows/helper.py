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

    def _transform(self, x, compute_y, compute_log_det):
        y, log_det = quadratic_transform(tfops, x, self.a, self.b)
        if not compute_y:
            y = None
        if not compute_log_det:
            log_det = None
        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        x, log_det = quadratic_inverse_transform(tfops, y, self.a, self.b)
        if not compute_x:
            x = None
        if not compute_log_det:
            log_det = None
        return x, log_det


def invertible_flow_standard_check(self, flow, session, x, feed_dict=None,
                                   rtol=1e-5):
    x = tf.convert_to_tensor(x)
    self.assertTrue(flow.explicitly_invertible)

    # test mapping from x -> y -> x
    y, log_det_y = flow.transform(x)
    x2, log_det_x = flow.inverse_transform(y)

    np.testing.assert_allclose(
        *session.run([x, x2], feed_dict=feed_dict), rtol=rtol)
    np.testing.assert_allclose(
        *session.run([log_det_y, -log_det_x], feed_dict=feed_dict), rtol=rtol)
