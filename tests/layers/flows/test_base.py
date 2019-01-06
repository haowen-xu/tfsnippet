import pytest

from tfsnippet.layers import *
from tests.layers.flows.helper import *


class FlowTestCase(tf.test.TestCase):

    def test_with_quadratic_flow(self):
        # test transform
        flow = QuadraticFlow(2., 5.)
        self.assertTrue(flow.explicitly_invertible)

        test_x = np.arange(12, dtype=np.float32) + 1.
        test_y, test_log_det = quadratic_transform(npyops, test_x, 2., 5.)

        self.assertFalse(flow._has_built)
        y, log_det_y = flow.transform(tf.constant(test_x))
        self.assertTrue(flow._has_built)

        with self.test_session() as sess:
            np.testing.assert_allclose(sess.run(y), test_y)
            np.testing.assert_allclose(sess.run(log_det_y), test_log_det)
            invertible_flow_standard_check(self, flow, sess, test_x)

        # test apply
        flow = QuadraticFlow(2., 5.)
        self.assertFalse(flow._has_built)
        y = flow.apply(tf.constant(test_x))
        self.assertTrue(flow._has_built)

        with self.test_session() as sess:
            np.testing.assert_allclose(sess.run(y), test_y)

    def test_error(self):
        class _Flow(BaseFlow):
            @property
            def explicitly_invertible(self):
                return False
        with pytest.raises(RuntimeError,
                           match='The flow is not explicitly invertible'):
            _ = _Flow().inverse_transform(tf.constant(0.))

        # specify neither `compute_y` nor `compute_log_det` would cause error
        flow = QuadraticFlow(2., 5.)
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

        # test `inverse_transform` should only be called after built
        flow = QuadraticFlow(2., 5.)
        self.assertFalse(flow._has_built)
        with pytest.raises(
                RuntimeError, match='`inverse_transform` cannot be called '
                                    'before the flow has been built'):
            _ = flow.inverse_transform(tf.constant(0.))

    def test_shape_assertion(self):
        class _Flow(BaseFlow):
            @property
            def explicitly_invertible(self):
                return True

            def _build(self, input=None):
                pass

            def _transform(self, x, compute_y, compute_log_det):
                return x, x + 1.

            def _inverse_transform(self, y, compute_x, compute_log_det):
                return y, y - 1.

        with self.test_session() as sess:
            flow = _Flow(value_ndims=1)

            # shape assertions in transform
            with pytest.raises(Exception,
                               match='The shape of `log_det` does not match '
                                     'the shape of `input`'):
                sess.run(flow.transform(tf.zeros([3, 4])))
            with pytest.raises(Exception,
                               match='`x.ndims` must be known and >= '
                                     '`value_ndims`'):
                sess.run(flow.transform(tf.constant(0.)))

            # shape assertions in inverse_transform
            with pytest.raises(Exception,
                               match='The shape of `log_det` does not match '
                                     'the shape of `input`'):
                sess.run(flow.inverse_transform(tf.zeros([3, 4])))
            with pytest.raises(Exception,
                               match='`y.ndims` must be known and >= '
                                     '`value_ndims`'):
                sess.run(flow.inverse_transform(tf.constant(0.)))

            # shape assertions in build
            flow = _Flow(value_ndims=1)
            with pytest.raises(Exception,
                               match='`input.ndims` must be known and >= '
                                     '`value_ndims`'):
                sess.run(flow.build(tf.constant(0.)))


class MultiLayerQuadraticFlow(MultiLayerFlow):

    def __init__(self, n_layers):
        super(MultiLayerQuadraticFlow, self).__init__(n_layers=n_layers)
        self._flows = []

        with tf.variable_scope(None, default_name='MultiLayerQuadraticFlow'):
            for layer_id in range(self.n_layers):
                self._flows.append(
                    QuadraticFlow(layer_id + 1, layer_id * 2 + 1))

    def _build(self, input=None):
        pass

    @property
    def explicitly_invertible(self):
        return True

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        flow = self._flows[layer_id]
        return flow.transform(x, compute_y, compute_log_det)

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        flow = self._flows[layer_id]
        return flow.inverse_transform(y, compute_x, compute_log_det)


class MultiLayerFlowTestCase(tf.test.TestCase):

    def test_with_multi_layer_quadratic_flow(self):
        n_layers = 3
        flow = MultiLayerQuadraticFlow(n_layers)

        # test properties
        self.assertEqual(flow.n_layers, n_layers)
        self.assertTrue(flow.explicitly_invertible)

        # test get parameters
        for i in range(n_layers):
            f = flow._flows[i]
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
