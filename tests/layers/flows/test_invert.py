import numpy as np
import pytest
import tensorflow as tf

from tests.layers.flows.helper import (QuadraticFlow, quadratic_transform,
                                       npyops, invertible_flow_standard_check)
from tfsnippet import FlowDistribution, Normal
from tfsnippet.layers import InvertFlow, PlanarNormalizingFlow, BaseFlow


class InvertFlowTestCase(tf.test.TestCase):

    def test_invert_flow(self):
        with self.test_session() as sess:
            # test invert a normal flow
            flow = QuadraticFlow(2., 5.)
            inv_flow = flow.invert()

            self.assertIsInstance(inv_flow, InvertFlow)
            self.assertEqual(inv_flow.x_value_ndims, 0)
            self.assertEqual(inv_flow.y_value_ndims, 0)
            self.assertFalse(inv_flow.require_batch_dims)

            test_x = np.arange(12, dtype=np.float32) + 1.
            test_y, test_log_det = quadratic_transform(npyops, test_x, 2., 5.)

            self.assertFalse(flow._has_built)
            y, log_det_y = inv_flow.inverse_transform(tf.constant(test_x))
            self.assertTrue(flow._has_built)

            np.testing.assert_allclose(sess.run(y), test_y)
            np.testing.assert_allclose(sess.run(log_det_y), test_log_det)
            invertible_flow_standard_check(self, inv_flow, sess, test_y)

            # test invert an InvertFlow
            inv_inv_flow = inv_flow.invert()
            self.assertIs(inv_inv_flow, flow)

            # test use with FlowDistribution
            normal = Normal(mean=1., std=2.)
            inv_flow = QuadraticFlow(2., 5.).invert()
            distrib = FlowDistribution(normal, inv_flow)
            distrib_log_det = distrib.log_prob(test_x)
            np.testing.assert_allclose(
                *sess.run([distrib_log_det,
                           normal.log_prob(test_y) + test_log_det])
            )

    def test_property(self):
        class _Flow(BaseFlow):
            @property
            def explicitly_invertible(self):
                return True

        inv_flow = _Flow(x_value_ndims=2, y_value_ndims=3,
                         require_batch_dims=True).invert()
        self.assertTrue(inv_flow.require_batch_dims)
        self.assertEqual(inv_flow.x_value_ndims, 3)
        self.assertEqual(inv_flow.y_value_ndims, 2)

    def test_errors(self):
        with pytest.raises(ValueError, match='`flow` must be an explicitly '
                                             'invertible flow'):
            _ = InvertFlow(object())

        with pytest.raises(ValueError, match='`flow` must be an explicitly '
                                             'invertible flow'):
            _ = PlanarNormalizingFlow().invert()
