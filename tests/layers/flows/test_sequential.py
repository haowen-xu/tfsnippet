import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.layers import SequentialFlow, BaseFlow
from tests.layers.flows.test_base import MultiLayerQuadraticFlow
from tests.layers.flows.helper import (QuadraticFlow,
                                       invertible_flow_standard_check)


class SequentialFlowTestCase(tf.test.TestCase):

    def test_errors(self):
        class _Flow(BaseFlow):
            pass

        with pytest.raises(TypeError, match='`flows` must not be empty'):
            _ = SequentialFlow([])

        with pytest.raises(
                TypeError, match='The 0-th flow in `flows` is not an instance '
                                 'of `Flow`: 123'):
            _ = SequentialFlow([123])

        with pytest.raises(
                TypeError, match='`value_ndims` of the 1-th flow in `flows` '
                                 'does not agree with the first flow: 2 vs 1'):
            _ = SequentialFlow([
                _Flow(value_ndims=1),
                _Flow(value_ndims=2),
            ])

    def test_sequential_with_quadratic_flows(self):
        n_layers = 3
        flow1 = MultiLayerQuadraticFlow(n_layers)
        flow2 = SequentialFlow([
            QuadraticFlow(i + 1., i * 2. + 1.)
            for i in range(n_layers)
        ])
        self.assertTrue(flow2.explicitly_invertible)
        self.assertEqual(len(flow2.flows), n_layers)
        for i in range(n_layers):
            self.assertEqual(flow2.flows[i].a, i + 1.)
            self.assertEqual(flow2.flows[i].b, i * 2. + 1.)

        x = tf.range(12, dtype=tf.float32) + 1.
        previous_log_det = 10. * np.random.normal(size=x.shape)

        with self.test_session() as sess:
            invertible_flow_standard_check(self, flow2, sess, x)

            # transform, without previous_log_det
            y1, log_det_y1 = flow1.transform(x)
            y2, log_det_y2 = flow2.transform(x)
            np.testing.assert_allclose(*sess.run([y1, y2]))
            np.testing.assert_allclose(*sess.run([log_det_y1, log_det_y2]))

            # transform, with previous_log_det
            y1, log_det_y1 = flow1.transform(
                x, previous_log_det=previous_log_det)
            y2, log_det_y2 = flow2.transform(
                x, previous_log_det=previous_log_det)
            np.testing.assert_allclose(*sess.run([y1, y2]))
            np.testing.assert_allclose(*sess.run([log_det_y1, log_det_y2]))

            # inverse transform, without previous_log_det
            x1, log_det_x1 = flow1.inverse_transform(y1)
            x2, log_det_x2 = flow1.inverse_transform(y2)
            np.testing.assert_allclose(*sess.run([x1, x2]))
            np.testing.assert_allclose(*sess.run([log_det_x1, log_det_x2]))

            # inverse transform, with previous_log_det
            x1, log_det_x1 = flow1.inverse_transform(
                y1, previous_log_det=previous_log_det)
            x2, log_det_x2 = flow1.inverse_transform(
                y2, previous_log_det=previous_log_det)
            np.testing.assert_allclose(*sess.run([x1, x2]))
            np.testing.assert_allclose(*sess.run([log_det_x1, log_det_x2]))

    def test_property(self):
        class _Flow(BaseFlow):
            @property
            def explicitly_invertible(self):
                return False

        flow = SequentialFlow([
            QuadraticFlow(2., 3.),
            _Flow(),
        ])
        self.assertFalse(flow.explicitly_invertible)
