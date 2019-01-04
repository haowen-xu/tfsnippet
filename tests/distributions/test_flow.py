import pytest
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.distributions import Normal, Categorical, FlowDistribution
from tfsnippet.utils import get_static_shape
from tests.layers.flows.helper import QuadraticFlow


class FlowDistributionTestCase(tf.test.TestCase):

    def test_errors(self):
        normal = Normal(mean=0., std=1.)
        with pytest.raises(TypeError,
                           match='`flow` is not an instance of `Flow`: 123'):
            _ = FlowDistribution(normal, 123)

        flow = QuadraticFlow(2., 5., dtype=tf.float32)
        with pytest.raises(ValueError,
                           match='cannot be transformed by a flow, because '
                                 'it is not continuous'):
            _ = FlowDistribution(Categorical(logits=[0., 1., 2.]), flow)
        with pytest.raises(ValueError,
                           match='cannot be transformed by a flow, because '
                                 'its data type is not float'):
            _ = FlowDistribution(Mock(normal, dtype=tf.int32), flow)

        distrib = FlowDistribution(normal, flow)
        with pytest.raises(RuntimeError,
                           match='`FlowDistribution` requires `compute_prob` '
                                 'not to be False'):
            _ = distrib.sample(compute_density=False)

    def test_property(self):
        normal = Normal(mean=[0., 1., 2.], std=1.)
        flow = QuadraticFlow(2., 5., dtype=tf.float64)
        distrib = FlowDistribution(normal, flow)

        self.assertIs(distrib.flow, flow)
        self.assertIs(distrib.distribution, normal)
        self.assertEqual(distrib.dtype, tf.float64)
        self.assertTrue(distrib.is_continuous)
        self.assertTrue(distrib.is_reparameterized)
        self.assertEqual(distrib.get_value_shape(), normal.get_value_shape())
        self.assertEqual(distrib.get_batch_shape(), normal.get_batch_shape())
        with self.test_session() as sess:
            np.testing.assert_equal(
                *sess.run([distrib.value_shape, normal.value_shape]))
            np.testing.assert_equal(
                *sess.run([distrib.batch_shape, normal.batch_shape]))

        # test is_reparameterized = False
        normal = Normal(mean=[0., 1., 2.], std=1., is_reparameterized=False)
        distrib = FlowDistribution(normal, flow)
        self.assertFalse(distrib.is_reparameterized)

    def test_sample(self):
        tf.set_random_seed(123456)

        mean = tf.constant([0., 1., 2.], dtype=tf.float64)
        normal = Normal(mean=mean, std=tf.constant(1., dtype=tf.float64))
        flow = QuadraticFlow(2., 5., dtype=tf.float64)
        distrib = FlowDistribution(normal, flow)

        # test ordinary sample, is_reparameterized = None
        y = distrib.sample(n_samples=5)
        self.assertTrue(y.is_reparameterized)
        grad = tf.gradients(y * 1., mean)[0]
        self.assertIsNotNone(grad)
        self.assertEqual(get_static_shape(y), (5, 3))
        self.assertIsNotNone(y._self_log_prob)

        x, log_det = flow.inverse_transform(y)
        log_py = normal.log_prob(x) + log_det

        with self.test_session() as sess:
            np.testing.assert_allclose(
                *sess.run([log_py, y.log_prob()]), rtol=1e-5)

        # test stop gradient sample, is_reparameterized = False
        y = distrib.sample(n_samples=5, is_reparameterized=False)
        self.assertFalse(y.is_reparameterized)
        grad = tf.gradients(y * 1., mean)[0]
        self.assertIsNone(grad)

    def test_log_prob(self):
        mean = tf.constant([0., 1., 2.], dtype=tf.float64)
        normal = Normal(mean=mean, std=tf.constant(1., dtype=tf.float64))
        flow = QuadraticFlow(2., 5., dtype=tf.float64)
        distrib = FlowDistribution(normal, flow)

        y = tf.constant([1., -1., 2.])
        x, log_det = flow.inverse_transform(y)
        log_py = normal.log_prob(x) + log_det

        with self.test_session() as sess:
            np.testing.assert_allclose(
                *sess.run([log_py, distrib.log_prob(y)]), rtol=1e-5)
