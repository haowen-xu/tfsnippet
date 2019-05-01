import pytest
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet import StochasticTensor, FlowDistributionDerivedTensor
from tfsnippet.distributions import Normal, Categorical, FlowDistribution
from tfsnippet.layers import ReshapeFlow
from tfsnippet.utils import get_static_shape
from tests.layers.flows.helper import QuadraticFlow


class FlowDistributionTestCase(tf.test.TestCase):

    def test_errors(self):
        # errors in constructor
        normal = Normal(mean=tf.zeros([3]), std=1.)
        with pytest.raises(TypeError, match='`flow` is not an instance of '
                                            '`BaseFlow`: 123'):
            _ = FlowDistribution(normal, 123)

        flow = QuadraticFlow(2., 5., value_ndims=1)
        with pytest.raises(ValueError,
                           match='cannot be transformed by a flow, because '
                                 'it is not continuous'):
            _ = FlowDistribution(Categorical(logits=[0., 1., 2.]), flow)

        with pytest.raises(ValueError,
                           match='cannot be transformed by a flow, because '
                                 'its data type is not float'):
            _ = FlowDistribution(Mock(normal, dtype=tf.int32), flow)

        with pytest.raises(ValueError,
                           match='cannot be transformed by flow .*, because '
                                 'distribution.value_ndims is larger than '
                                 'flow.x_value_ndims'):
            _ = FlowDistribution(Mock(normal, value_ndims=2), flow)

        # errors in sample
        distrib = FlowDistribution(normal, flow)
        with pytest.raises(RuntimeError,
                           match='`FlowDistribution` requires `compute_prob` '
                                 'not to be False'):
            _ = distrib.sample(compute_density=False)

    def test_property(self):
        normal = Normal(mean=tf.constant([0., 1., 2.], dtype=tf.float64),
                        std=tf.constant(1., dtype=tf.float64))
        flow = QuadraticFlow(2., 5., value_ndims=1)
        distrib = FlowDistribution(normal, flow)

        self.assertIs(distrib.flow, flow)
        self.assertIs(distrib.base_distribution, normal)
        self.assertEqual(distrib.dtype, tf.float64)
        self.assertTrue(distrib.is_continuous)
        self.assertTrue(distrib.is_reparameterized)
        self.assertEqual(distrib.value_ndims, 1)

        # self.assertEqual(distrib.get_value_shape(), normal.get_value_shape())
        # self.assertEqual(distrib.get_batch_shape(), normal.get_batch_shape())
        # with self.test_session() as sess:
        #     np.testing.assert_equal(
        #         *sess.run([distrib.value_shape, normal.value_shape]))
        #     np.testing.assert_equal(
        #         *sess.run([distrib.batch_shape, normal.batch_shape]))

        # test is_reparameterized = False
        normal = Normal(mean=[0., 1., 2.], std=1., is_reparameterized=False)
        distrib = FlowDistribution(normal, flow)
        self.assertFalse(distrib.is_reparameterized)

        # test y_value_ndims = 2
        distrib = FlowDistribution(normal, ReshapeFlow(1, [-1, 1]))
        self.assertEqual(distrib.value_ndims, 2)

    def test_sample(self):
        tf.set_random_seed(123456)

        mean = tf.constant([0., 1., 2.], dtype=tf.float64)
        normal = Normal(mean=mean, std=tf.constant(1., dtype=tf.float64))
        flow = QuadraticFlow(2., 5.)
        distrib = FlowDistribution(normal, flow)

        # test ordinary sample, is_reparameterized = None
        y = distrib.sample(n_samples=5)
        self.assertTrue(y.is_reparameterized)
        self.assertIsInstance(y.flow_origin, StochasticTensor)
        self.assertIs(y.flow_origin.distribution, normal)

        grad = tf.gradients(y * 1., mean)[0]
        self.assertIsNotNone(grad)
        self.assertEqual(get_static_shape(y), (5, 3))
        self.assertIsNotNone(y._self_log_prob)

        log_prob = y.log_prob()
        self.assertIsInstance(log_prob, FlowDistributionDerivedTensor)
        self.assertIs(log_prob.flow_origin, y.flow_origin)

        prob = y.prob()
        self.assertIsInstance(prob, FlowDistributionDerivedTensor)
        self.assertIs(prob.flow_origin, y.flow_origin)

        x, log_det = flow.inverse_transform(y)
        log_py = normal.log_prob(x) + log_det
        py = tf.exp(log_py)

        with self.test_session() as sess:
            np.testing.assert_allclose(
                *sess.run([y.flow_origin, x]), rtol=1e-5)
            np.testing.assert_allclose(
                *sess.run([log_py, log_prob]), rtol=1e-5)
            np.testing.assert_allclose(
                *sess.run([py, prob]), rtol=1e-5)

        # test stop gradient sample, is_reparameterized = False
        y = distrib.sample(n_samples=5, is_reparameterized=False)
        self.assertFalse(y.is_reparameterized)
        grad = tf.gradients(y * 1., mean)[0]
        self.assertIsNone(grad)

    def test_sample_value_and_group_ndims(self):
        tf.set_random_seed(123456)

        mean = tf.constant([0., 1., 2.], dtype=tf.float64)
        normal = Normal(mean=mean, std=tf.constant(1., dtype=tf.float64))

        with self.test_session() as sess:
            # test value_ndims = 0, group_ndims = 1
            flow = QuadraticFlow(2., 5.)
            distrib = FlowDistribution(normal, flow)
            self.assertEqual(distrib.value_ndims, 0)

            y = distrib.sample(n_samples=5, group_ndims=1)
            self.assertTupleEqual(get_static_shape(y), (5, 3))
            x, log_det = flow.inverse_transform(y)
            self.assertTupleEqual(get_static_shape(x), (5, 3))
            self.assertTupleEqual(get_static_shape(log_det), (5, 3))
            log_py = tf.reduce_sum(normal.log_prob(x) + log_det, axis=-1)

            np.testing.assert_allclose(*sess.run([y.log_prob(), log_py]),
                                       rtol=1e-5)

            # test value_ndims = 1, group_ndims = 0
            flow = QuadraticFlow(2., 5., value_ndims=1)
            distrib = FlowDistribution(normal, flow)
            self.assertEqual(distrib.value_ndims, 1)

            y = distrib.sample(n_samples=5, group_ndims=0)
            self.assertTupleEqual(get_static_shape(y), (5, 3))
            x, log_det = flow.inverse_transform(y)
            self.assertTupleEqual(get_static_shape(x), (5, 3))
            self.assertTupleEqual(get_static_shape(log_det), (5,))
            log_py = log_det + tf.reduce_sum(normal.log_prob(x), axis=-1)

            np.testing.assert_allclose(*sess.run([y.log_prob(), log_py]),
                                       rtol=1e-5)

            # test value_ndims = 1, group_ndims = 1
            flow = QuadraticFlow(2., 5., value_ndims=1)
            distrib = FlowDistribution(normal, flow)
            self.assertEqual(distrib.value_ndims, 1)

            y = distrib.sample(n_samples=5, group_ndims=1)
            self.assertTupleEqual(get_static_shape(y), (5, 3))
            x, log_det = flow.inverse_transform(y)
            self.assertTupleEqual(get_static_shape(x), (5, 3))
            self.assertTupleEqual(get_static_shape(log_det), (5,))
            log_py = tf.reduce_sum(
                log_det + tf.reduce_sum(normal.log_prob(x), axis=-1))

            np.testing.assert_allclose(*sess.run([y.log_prob(), log_py]),
                                       rtol=1e-5)

    def test_log_prob(self):
        mean = tf.constant([0., 1., 2.], dtype=tf.float64)
        normal = Normal(mean=mean, std=tf.constant(1., dtype=tf.float64))
        flow = QuadraticFlow(2., 5.)
        flow.build(tf.constant(0., dtype=tf.float64))
        distrib = FlowDistribution(normal, flow)

        y = tf.constant([1., -1., 2.], dtype=tf.float64)
        x, log_det = flow.inverse_transform(y)
        log_py = normal.log_prob(x) + log_det
        py = tf.exp(log_py)

        log_prob = distrib.log_prob(y)
        self.assertIsInstance(log_prob, FlowDistributionDerivedTensor)
        self.assertIsInstance(log_prob.flow_origin, StochasticTensor)
        self.assertIs(log_prob.flow_origin.distribution, normal)

        prob = distrib.prob(y)
        self.assertIsInstance(prob, FlowDistributionDerivedTensor)
        self.assertIsInstance(prob.flow_origin, StochasticTensor)
        self.assertIs(prob.flow_origin.distribution, normal)

        with self.test_session() as sess:
            np.testing.assert_allclose(
                *sess.run([log_prob.flow_origin, x]), rtol=1e-5)
            np.testing.assert_allclose(
                *sess.run([log_py, log_prob]), rtol=1e-5)
            np.testing.assert_allclose(
                *sess.run([py, prob]), rtol=1e-5)

    def test_log_prob_value_and_group_ndims(self):
        tf.set_random_seed(123456)

        mean = tf.constant([0., 1., 2.], dtype=tf.float64)
        normal = Normal(mean=mean, std=tf.constant(1., dtype=tf.float64))
        y = tf.random_normal(shape=[2, 5, 3], dtype=tf.float64)

        with self.test_session() as sess:
            # test value_ndims = 0, group_ndims = 1
            flow = QuadraticFlow(2., 5.)
            flow.build(tf.zeros([2, 5, 3], dtype=tf.float64))
            distrib = FlowDistribution(normal, flow)
            self.assertEqual(distrib.value_ndims, 0)

            x, log_det = flow.inverse_transform(y)
            self.assertTupleEqual(get_static_shape(x), (2, 5, 3))
            self.assertTupleEqual(get_static_shape(log_det), (2, 5, 3))
            log_py = tf.reduce_sum(normal.log_prob(x) + log_det, axis=-1)

            np.testing.assert_allclose(
                *sess.run([distrib.log_prob(y, group_ndims=1), log_py]),
                rtol=1e-5
            )

            # test value_ndims = 1, group_ndims = 0
            flow = QuadraticFlow(2., 5., value_ndims=1)
            flow.build(tf.zeros([2, 5, 3], dtype=tf.float64))
            distrib = FlowDistribution(normal, flow)
            self.assertEqual(distrib.value_ndims, 1)

            x, log_det = flow.inverse_transform(y)
            self.assertTupleEqual(get_static_shape(x), (2, 5, 3))
            self.assertTupleEqual(get_static_shape(log_det), (2, 5))
            log_py = normal.log_prob(x, group_ndims=1) + log_det

            np.testing.assert_allclose(
                *sess.run([distrib.log_prob(y, group_ndims=0), log_py]),
                rtol=1e-5
            )

            # test value_ndims = 1, group_ndims = 2
            flow = QuadraticFlow(2., 5., value_ndims=1)
            flow.build(tf.zeros([2, 5, 3], dtype=tf.float64))
            distrib = FlowDistribution(normal, flow)
            self.assertEqual(distrib.value_ndims, 1)

            x, log_det = flow.inverse_transform(y)
            self.assertTupleEqual(get_static_shape(x), (2, 5, 3))
            self.assertTupleEqual(get_static_shape(log_det), (2, 5))
            log_py = tf.reduce_sum(
                log_det + tf.reduce_sum(normal.log_prob(x), axis=-1))

            np.testing.assert_allclose(
                *sess.run([distrib.log_prob(y, group_ndims=2), log_py]),
                rtol=1e-5
            )
