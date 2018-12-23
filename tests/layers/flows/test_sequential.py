import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.layers import SequentialFlow, BaseFlow
from tests.layers.flows.test_base import MultiLayerQuadraticFlow
from tests.layers.flows.helper import QuadraticFlow, invertible_flow_standard_check


class SequentialFlowTestCase(tf.test.TestCase):

    def test_errors(self):
        with pytest.raises(TypeError, match='`flows` must not be empty'):
            _ = SequentialFlow([])
        with pytest.raises(
                TypeError, match='The 0-th flow in `flows` is not an instance '
                                 'of `Flow`: 123'):
            _ = SequentialFlow([123])

    def test_sequential_with_quadratic_flows(self):
        n_layers = 3
        flow1 = MultiLayerQuadraticFlow(n_layers)
        flow2 = SequentialFlow([
            QuadraticFlow(i + 1., i * 2. + 1., dtype=tf.float32)
            for i in range(n_layers)
        ])
        self.assertTrue(flow2.explicitly_invertible)
        self.assertEqual(flow2.dtype, tf.float32)
        self.assertEqual(len(flow2.flows), n_layers)
        for i in range(n_layers):
            self.assertEqual(flow2.flows[i].a, i + 1.)
            self.assertEqual(flow2.flows[i].b, i * 2. + 1.)

        x = tf.range(12, dtype=tf.float32) + 1.
        with self.test_session() as sess:
            invertible_flow_standard_check(self, flow2, sess, x)

            y1, log_det_y1 = flow1.transform(x)
            y2, log_det_y2 = flow2.transform(x)
            np.testing.assert_allclose(*sess.run([y1, y2]))
            np.testing.assert_allclose(*sess.run([log_det_y1, log_det_y2]))

            x1, log_det_x1 = flow1.inverse_transform(y1)
            x2, log_det_x2 = flow1.inverse_transform(y2)
            np.testing.assert_allclose(*sess.run([x1, x2]))
            np.testing.assert_allclose(*sess.run([log_det_x1, log_det_x2]))

    def test_property(self):
        class _Flow(BaseFlow):
            @property
            def explicitly_invertible(self):
                return False

        flow = SequentialFlow([
            QuadraticFlow(2., 3., dtype=tf.float32),
            _Flow(dtype=tf.float64),
        ])
        self.assertFalse(flow.explicitly_invertible)
        self.assertEqual(flow.dtype, tf.float64)
