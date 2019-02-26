import functools

import pytest
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.layers import *


class InvertibleActivationTestCase(tf.test.TestCase):

    def test_call(self):
        x, y, log_det = object(), object(), object()
        a = InvertibleActivation()
        a.transform = Mock(return_value=(y, log_det))

        y2 = a(x)
        self.assertIs(y2, y)

        self.assertEqual(a.transform.call_args, (
            (),
            {
                'x': x,
                'compute_y': True,
                'compute_log_det': False,
                'name': 'invertible_activation',
            }
        ))

    def test_transform(self):
        with pytest.raises(ValueError,
                           match='At least one of `compute_y` and '
                                 '`compute_log_det` should be True'):
            _ = InvertibleActivation().transform(
                tf.zeros([]), compute_y=False, compute_log_det=False)
        with pytest.raises(ValueError, match='`value_ndims` must be >= 0'):
            _ = InvertibleActivation().transform(
                tf.zeros([]), value_ndims=-1)

        # other functions are tested in test_invertible_activation_flow

    def test_inverse_transform(self):
        with pytest.raises(ValueError,
                           match='At least one of `compute_x` and '
                                 '`compute_log_det` should be True'):
            _ = InvertibleActivation().inverse_transform(
                tf.zeros([]), compute_x=False, compute_log_det=False)
        with pytest.raises(ValueError, match='`value_ndims` must be >= 0'):
            _ = InvertibleActivation().inverse_transform(
                tf.zeros([]), value_ndims=-1)

        # other functions are tested in test_invertible_activation_flow

    def test_as_flow(self):
        a = InvertibleActivation()
        f = a.as_flow(value_ndims=3)
        self.assertIs(f.activation, a)
        self.assertEqual(f.value_ndims, 3)
        self.assertEqual(f.x_value_ndims, 3)
        self.assertEqual(f.y_value_ndims, 3)


class InvertibleActivationFlowTestCase(tf.test.TestCase):

    def test_invertible_activation_flow(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        with pytest.raises(TypeError, match='`activation` must be an instance '
                                            'of `InvertibleActivation`'):
            _ = InvertibleActivationFlow(activation=object(), value_ndims=0)

        leaky_relu = LeakyReLU()
        f = InvertibleActivationFlow(leaky_relu, value_ndims=1)
        self.assertTrue(f.explicitly_invertible)
        self.assertEqual(f.value_ndims, 1)
        self.assertEqual(f.x_value_ndims, 1)
        self.assertEqual(f.y_value_ndims, 1)
        self.assertIs(f.activation, leaky_relu)

        x = np.random.normal(size=[2, 3, 4]).astype(np.float32)
        y = np.where(x < 0, x * 0.2, x)
        log_det = np.where(
            x < 0,
            (np.log(0.2).astype(np.float32) *
             np.ones([2, 3, 4], dtype=np.float32)),
            np.zeros([2, 3, 4], dtype=np.float32)
        )
        log_det = np.sum(log_det, axis=-1)

        with self.test_session() as sess:
            y_out, log_det_out = sess.run(f.transform(x))
            x2_out, log_det2_out = sess.run(f.inverse_transform(y))

            assert_allclose(y_out, y)
            assert_allclose(log_det_out, log_det)
            assert_allclose(x2_out, x)
            assert_allclose(log_det2_out, -log_det)
