import pytest

from tfsnippet.layers import *
from tests.layers.flows.helper import *
from tfsnippet.layers.flows import FeatureMappingFlow


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
        # test invertibility
        class _Flow(BaseFlow):
            @property
            def explicitly_invertible(self):
                return False
        with pytest.raises(RuntimeError,
                           match='The flow is not explicitly invertible'):
            _ = _Flow().inverse_transform(tf.constant(0.))

        # test build without input will cause error
        with pytest.raises(ValueError,
                           match='`input` is required to build _Flow'):
            _ = _Flow().build(None)

        # specify neither `compute_y` nor `compute_log_det` will cause error
        flow = QuadraticFlow(2., 5.)
        with pytest.raises(
                ValueError, match='At least one of `compute_y` and '
                                  '`compute_log_det` should be True'):
            _ = flow.transform(
                tf.constant(0.), compute_y=False, compute_log_det=False)
        with pytest.raises(
                ValueError, match='At least one of `compute_x` and '
                                  '`compute_log_det` should be True'):
            _ = flow.inverse_transform(
                tf.constant(0.), compute_x=False, compute_log_det=False)

        # specify `previous_log_det` without `compute_log_det` will cause error
        with pytest.raises(
                ValueError, match='`previous_log_det` is specified but '
                                  '`compute_log_det` is False'):
            _ = flow.transform(tf.constant(0.), compute_log_det=False,
                               previous_log_det=tf.constant(1.))
        with pytest.raises(
                ValueError, match='`previous_log_det` is specified but '
                                  '`compute_log_det` is False'):
            _ = flow.inverse_transform(tf.constant(0.), compute_log_det=False,
                                       previous_log_det=tf.constant(1.))

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
            # shape assertions in transform
            flow = _Flow(value_ndims=1)
            with pytest.raises(Exception,
                               match='`x.ndims` must be known and >= '
                                     '`value_ndims`'):
                sess.run(flow.transform(tf.constant(0.)))

            # shape assertions in transform, require_batch_ndims is True
            flow = _Flow(value_ndims=1, require_batch_dims=True)
            with pytest.raises(Exception,
                               match=r'`x.ndims` must be known and >= '
                                     r'`value_ndims \+ 1`'):
                sess.run(flow.transform(tf.constant([0.])))

            # shape assertions in inverse_transform
            flow = _Flow(value_ndims=1)
            flow.build(tf.zeros([2, 3]))
            with pytest.raises(Exception,
                               match='`y.ndims` must be known and >= '
                                     '`value_ndims`'):
                sess.run(flow.inverse_transform(tf.constant(0.)))

            # shape assertions in transform, require_batch_ndims is True
            flow = _Flow(value_ndims=1, require_batch_dims=True)
            flow.build(tf.zeros([2, 3]))
            with pytest.raises(Exception,
                               match=r'`y.ndims` must be known and >= '
                                     r'`value_ndims \+ 1`'):
                sess.run(flow.inverse_transform(tf.constant([0.])))

            # shape assertions in build
            flow = _Flow(value_ndims=1)
            with pytest.raises(Exception,
                               match='`input.ndims` must be known and >= '
                                     '`value_ndims`'):
                sess.run(flow.build(tf.constant(0.)))

            flow = _Flow(value_ndims=1, require_batch_dims=True)
            with pytest.raises(Exception,
                               match=r'`input.ndims` must be known and >= '
                                     r'`value_ndims \+ 1`'):
                sess.run(flow.build(tf.constant([0.])))


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

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det,
                         previous_log_det):
        flow = self._flows[layer_id]
        return flow.transform(x, compute_y, compute_log_det, previous_log_det)

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det,
                                 previous_log_det):
        flow = self._flows[layer_id]
        return flow.inverse_transform(
            y, compute_x, compute_log_det, previous_log_det)


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
        previous_log_det = \
            10. * np.random.normal(size=test_log_det.shape).astype(np.float32)

        y, log_det_y = flow.transform(tf.constant(test_x))
        _, log_det_y2 = flow.transform(
            tf.constant(test_x), previous_log_det=previous_log_det)

        with self.test_session() as sess:
            np.testing.assert_allclose(sess.run(y), test_y)
            np.testing.assert_allclose(sess.run(log_det_y), test_log_det)
            np.testing.assert_allclose(sess.run(log_det_y2),
                                       test_log_det + previous_log_det)
            invertible_flow_standard_check(self, flow, sess, test_x)

    def test_errors(self):
        with pytest.raises(ValueError,
                           match='`n_layers` must be larger than 0'):
            _ = MultiLayerFlow(0)


class FeatureMappingFlowTestCase(tf.test.TestCase):

    def test_property(self):
        # test axis is integer
        flow = FeatureMappingFlow(axis=1, value_ndims=2)
        flow.build(tf.zeros([2, 3, 4]))
        self.assertEqual(flow.axis, -2)

        # test axis is tuple
        flow = FeatureMappingFlow(axis=[-1, 1], value_ndims=2)
        flow.build(tf.zeros([2, 3, 4]))
        self.assertEqual(flow.axis, (-2, -1))

    def test_errors(self):
        with pytest.raises(ValueError, match='`axis` must not be empty'):
            _ = FeatureMappingFlow(axis=(), value_ndims=1)

        with pytest.raises(ValueError, match='`axis` out of range, or not '
                                             'covered by `value_ndims`'):
            layer = FeatureMappingFlow(axis=-2, value_ndims=1)
            _ = layer.apply(tf.zeros([2, 3]))

        with pytest.raises(ValueError, match='Duplicated elements after '
                                             'resolving negative `axis` with '
                                             'respect to the `input`'):
            layer = FeatureMappingFlow(axis=[1, -1], value_ndims=1)
            _ = layer.apply(tf.zeros([2, 3]))

        with pytest.raises(ValueError, match='`axis` out of range, or not '
                                             'covered by `value_ndims`'):
            layer = FeatureMappingFlow(axis=-2, value_ndims=1)
            _ = layer.apply(tf.zeros([2, 3]))

        with pytest.raises(ValueError, match='The feature axis of `input` '
                                             'is not deterministic'):
            layer = FeatureMappingFlow(axis=(-1, -2), value_ndims=2)
            _ = layer.apply(tf.placeholder(dtype=tf.float32, shape=[None, 3]))

        with pytest.raises(ValueError, match='The feature axis of `input` '
                                             'is not deterministic'):
            layer = FeatureMappingFlow(axis=-2, value_ndims=2)
            _ = layer.apply(tf.placeholder(dtype=tf.float32, shape=[None, 3]))
