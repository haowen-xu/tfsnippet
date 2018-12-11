import numpy as np
import tensorflow as tf

from tfsnippet.flows import PlanarNormalizingFlow
from tfsnippet.utils import int_shape, ensure_variables_initialized


class PlanarNormalizingFlowTestCase(tf.test.TestCase):

    def test_basic(self):
        n_units = 5
        n_layers = 3
        flow = PlanarNormalizingFlow(n_units=n_units, n_layers=n_layers)

        self.assertFalse(flow.explicitly_invertible)
        self.assertEqual(flow.n_units, n_units)
        self.assertEqual(flow.n_layers, n_layers)

    def test_u_hat(self):
        n_units = 5
        n_layers = 3
        tf.set_random_seed(1234)
        flow = PlanarNormalizingFlow(n_units=n_units, n_layers=n_layers,
                                     dtype=tf.float64)

        for i in range(flow.n_layers):
            # ensure these parameters exist
            w, u, b, u_hat = flow.get_layer_params(i, ['w', 'u', 'b', 'u_hat'])

            for v in [w, u, b, u_hat]:
                self.assertIn('/_{}/'.format(i), v.name)

            # ensure these parameters have expected shapes
            self.assertEqual(int_shape(w), (1, n_units))
            self.assertEqual(int_shape(u), (1, n_units))
            self.assertEqual(int_shape(b), (1,))
            self.assertEqual(int_shape(u_hat), (1, n_units))

        with self.test_session() as sess:
            ensure_variables_initialized()

            for i in range(flow.n_layers):
                w, u, b, u_hat = sess.run(
                    flow.get_layer_params(i, ['w', 'u', 'b', 'u_hat']))
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
        n_layers = 3
        tf.set_random_seed(1234)
        flow = PlanarNormalizingFlow(n_units=n_units, n_layers=n_layers,
                                     dtype=tf.float64)
        x = np.arange(30, dtype=np.float64).reshape([2, 3, 5])

        with self.test_session() as sess:
            ensure_variables_initialized()

            # compute the ground-truth y and log_det
            y, log_det = x, 0
            for i in range(flow.n_layers):
                u, w, b = sess.run(
                    flow.get_layer_params(i, ['u_hat', 'w', 'b']))
                y, tmp = transform(y, u, w, b)
                log_det += tmp

            # check the flow-derived y and log_det
            y2, log_det2 = sess.run(
                flow.transform(tf.constant(x, dtype=tf.float64)))

            np.testing.assert_allclose(y2, y, rtol=1e-5)
            np.testing.assert_allclose(log_det2, log_det, rtol=1e-5)
