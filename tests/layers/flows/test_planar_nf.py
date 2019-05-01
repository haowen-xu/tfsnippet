import numpy as np
import tensorflow as tf

from tests.helper import assert_variables
from tfsnippet.layers import *
from tfsnippet.utils import get_static_shape, ensure_variables_initialized


class PlanarNormalizingFlowTestCase(tf.test.TestCase):

    def test_basic(self):
        flow = PlanarNormalizingFlow()
        self.assertFalse(flow.explicitly_invertible)

    def test_u_hat(self):
        n_units = 5
        tf.set_random_seed(1234)
        flow = PlanarNormalizingFlow(name='planar_nf')
        flow.apply(tf.random_normal(shape=[1, n_units]))

        # ensure these parameters exist
        w, b, u, u_hat = flow._w, flow._b, flow._u, flow._u_hat
        for v in [w, u, b, u_hat]:
            self.assertIn('planar_nf/', v.name)

        # ensure these parameters have expected shapes
        self.assertEqual(get_static_shape(w), (1, n_units))
        self.assertEqual(get_static_shape(u), (1, n_units))
        self.assertEqual(get_static_shape(b), (1,))
        self.assertEqual(get_static_shape(u_hat), (1, n_units))

        with self.test_session() as sess:
            ensure_variables_initialized()
            w, b, u, u_hat = sess.run([w, b, u, u_hat])
            m = lambda a: -1 + np.log(1 + np.exp(a))
            wu = np.dot(w, u.T)  # shape: [1]
            np.testing.assert_allclose(
                u + w * (m(wu) - wu) / np.sum(w ** 2),
                u_hat
            )

    def test_transform(self):
        def transform(x, u, w, b):
            wxb = np.dot(x, w.T) + b
            y = x + u * np.tanh(wxb)  # shape: [?, n_units]
            tanh_wxb = np.tanh(wxb)
            phi = (1 - tanh_wxb ** 2) * w  # shape: [?, n_units]
            log_det = np.log(np.abs(1 + np.dot(phi, u.T)))  # shape: [?, 1]
            return y, np.squeeze(log_det, -1)

        n_units = 5
        tf.set_random_seed(1234)
        flow = PlanarNormalizingFlow()
        flow.apply(tf.random_normal(shape=[1, n_units], dtype=tf.float64))
        x = np.arange(30, dtype=np.float64).reshape([2, 3, n_units])

        with self.test_session() as sess:
            ensure_variables_initialized()

            # compute the ground-truth y and log_det
            y = x
            w, b, u_hat = flow._w, flow._b, flow._u_hat
            u_hat, w, b = sess.run([u_hat, w, b])
            y, log_det = transform(y, u_hat, w, b)

            # check the flow-derived y and log_det
            y2, log_det2 = sess.run(
                flow.transform(tf.constant(x, dtype=tf.float64)))

            np.testing.assert_allclose(y2, y, rtol=1e-5)
            np.testing.assert_allclose(log_det2, log_det, rtol=1e-5)

    def test_planar_nf_vars(self):
        # test trainable
        with tf.Graph().as_default():
            _ = PlanarNormalizingFlow().apply(tf.zeros([2, 3]))
            assert_variables(['w', 'b', 'u'], trainable=True,
                             scope='planar_normalizing_flow',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test non-trainable
        with tf.Graph().as_default():
            _ = PlanarNormalizingFlow(trainable=False).apply(tf.zeros([2, 3]))
            assert_variables(['w', 'b', 'u'], trainable=False,
                             scope='planar_normalizing_flow',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

    def test_planar_normalizing_flows(self):
        # test single-layer flow
        flow = planar_normalizing_flows()
        self.assertIsInstance(flow, PlanarNormalizingFlow)
        self.assertEqual(flow.variable_scope.name, 'planar_normalizing_flow')

        # test multi-layer flows
        flow = planar_normalizing_flows(n_layers=5)
        self.assertIsInstance(flow, MultiLayerFlow)
        self.assertEqual(flow.n_layers, 5)
        for i, flow in enumerate(flow.flows):
            self.assertIsInstance(flow, PlanarNormalizingFlow)
            self.assertEqual(flow.variable_scope.name,
                             'planar_normalizing_flows/_{}'.format(i))
