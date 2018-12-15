import pytest

from tfsnippet.mathops import npyops
from tests.flows.helper import *


class FlowTestCase(tf.test.TestCase):

    def test_with_quadratic_flow(self):
        flow = QuadraticFlow(2., 5., dtype=tf.float64)

        # test properties
        self.assertEqual(flow.dtype, tf.float64)
        self.assertTrue(flow.explicitly_invertible)

        # test transform
        test_x = np.arange(12, dtype=np.float32) + 1.
        test_y, test_log_det = quadratic_transform(npyops, test_x, 2., 5.)
        y, log_det_y = flow.transform(tf.constant(test_x))
        self.assertEqual(y.dtype, tf.float64)

        with self.test_session() as sess:
            np.testing.assert_allclose(sess.run(y), test_y)
            np.testing.assert_allclose(sess.run(log_det_y), test_log_det)
            invertible_flow_standard_check(self, flow, sess, test_x)

    def test_error(self):
        with pytest.raises(TypeError, match='Expected a float dtype'):
            _ = Flow(dtype=tf.int64)

        class _Flow(Flow):
            @property
            def explicitly_invertible(self):
                return False
        with pytest.raises(RuntimeError,
                           match='The flow is not explicitly invertible'):
            _ = _Flow().inverse_transform(tf.constant(0.))

        flow = QuadraticFlow(2., 5., dtype=tf.float64)
        with pytest.raises(
                RuntimeError, match='At least one of `compute_y` and '
                                    '`compute_log_det` should be True'):
            _ = flow.transform(
                tf.constant(0.), compute_y=False, compute_log_det=False)
        with pytest.raises(
                RuntimeError, match='At least one of `compute_x` and '
                                    '`compute_log_det` should be True'):
            _ = flow.inverse_transform(
                tf.constant(0.), compute_x=False, compute_log_det=False)


class MultiLayerQuadraticFlow(MultiLayerFlow):

    def __init__(self, n_layers):
        super(MultiLayerQuadraticFlow, self).__init__(n_layers=n_layers)

    @property
    def explicitly_invertible(self):
        return True

    def _create_layer_params(self, layer_id):
        return {
            'a': tf.get_variable('a', dtype=tf.float32, shape=[1]),
            'flow': QuadraticFlow(
                layer_id + 1, layer_id * 2 + 1, dtype=tf.float32)
        }

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        flow = self.get_layer_params(layer_id, ['flow'])[0]
        return flow.transform(x, compute_y, compute_log_det)

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        flow = self.get_layer_params(layer_id, ['flow'])[0]
        return flow.inverse_transform(y, compute_x, compute_log_det)


class MultiLayerFlowTestCase(tf.test.TestCase):

    def test_with_multi_layer_quadratic_flow(self):
        n_layers = 3
        flow = MultiLayerQuadraticFlow(n_layers)

        # test properties
        self.assertEqual(flow.n_layers, n_layers)
        self.assertEqual(flow.dtype, tf.float32)
        self.assertTrue(flow.explicitly_invertible)

        # test get parameters
        for i in range(n_layers):
            a, f = flow.get_layer_params(i, ['a', 'flow'])
            self.assertIn('/_{}/'.format(i), a.name)
            self.assertEqual(f.a, i + 1.)
            self.assertEqual(f.b, i * 2 + 1.)

        # test transform
        test_x = np.arange(12, dtype=np.float32) + 1.
        test_y, test_log_det = test_x, 0
        for i in range(n_layers):
            test_y, tmp = quadratic_transform(
                npyops, test_y, i + 1., i * 2 + 1.)
            test_log_det += tmp
        y, log_det_y = flow.transform(tf.constant(test_x))
        with self.test_session() as sess:
            np.testing.assert_allclose(sess.run(y), test_y)
            np.testing.assert_allclose(sess.run(log_det_y), test_log_det)
            invertible_flow_standard_check(self, flow, sess, test_x)

    def test_errors(self):
        with pytest.raises(ValueError,
                           match='`n_layers` must be larger than 0'):
            _ = MultiLayerFlow(0)
