import functools

import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.layers import *


class LeakyReLUTestCase(tf.test.TestCase):

    def test_leaky_relu(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        np.random.seed(12345)
        x = np.random.normal(size=[11, 31, 51]).astype(np.float32)
        y = np.maximum(x * .2, x)
        log_det = np.where(
            x < 0,
            np.ones_like(x, dtype=np.float32) * np.log(.2).astype(np.float32),
            np.zeros_like(x, dtype=np.float32)
        )
        leaky_relu = LeakyReLU(alpha=.2)

        with self.test_session() as sess:
            # test value_ndims = 0
            y_out, log_det_out = sess.run(
                leaky_relu.transform(x))
            self.assertTupleEqual(y_out.shape, (11, 31, 51))
            self.assertTupleEqual(log_det_out.shape, (11, 31, 51))
            x2_out, log_det2_out = sess.run(
                leaky_relu.inverse_transform(y))
            assert_allclose(x2_out, x)
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)
            assert_allclose(log_det2_out, -log_det)

            # test value_ndims = 1
            y_out, log_det_out = sess.run(
                leaky_relu.transform(x, value_ndims=1))
            self.assertTupleEqual(y_out.shape, (11, 31, 51))
            self.assertTupleEqual(log_det_out.shape, (11, 31))
            x2_out, log_det2_out = sess.run(
                leaky_relu.inverse_transform(y, value_ndims=1))
            assert_allclose(x2_out, x)
            assert_allclose(y_out, y)
            assert_allclose(log_det_out, np.sum(log_det, axis=-1))
            assert_allclose(log_det2_out, -np.sum(log_det, axis=-1))

            # test call
            assert_allclose(sess.run(leaky_relu(x)), y)

        with pytest.raises(ValueError,
                           match='`alpha` must be a float number, '
                                 'and 0 < alpha < 1: got 0'):
            _ = LeakyReLU(alpha=0)
        with pytest.raises(ValueError, match='`alpha` must be a float number, '
                                             'and 0 < alpha < 1: got 1'):
            _ = LeakyReLU(alpha=1)
        with pytest.raises(ValueError, match='`alpha` must be a float number, '
                                             'and 0 < alpha < 1: got -1'):
            _ = LeakyReLU(alpha=-1)
        with pytest.raises(ValueError, match='`alpha` must be a float number, '
                                             'and 0 < alpha < 1: got 2'):
            _ = LeakyReLU(alpha=2)
