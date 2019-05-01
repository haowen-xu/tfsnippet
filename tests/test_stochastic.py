import pytest
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.stochastic import StochasticTensor
from tfsnippet.utils import TensorWrapper, register_tensor_wrapper_class


class _MyTensorWrapper(TensorWrapper):

    def __init__(self, wrapped):
        self._self_wrapped = wrapped

    @property
    def tensor(self):
        return self._self_wrapped


register_tensor_wrapper_class(_MyTensorWrapper)


class StochasticTensorTestCase(tf.test.TestCase):

    def test_equality(self):
        distrib = Mock(is_reparameterized=False)
        samples = tf.constant(0.)
        t = StochasticTensor(distrib, samples)
        self.assertEqual(t, t)
        self.assertEqual(hash(t), hash(t))
        self.assertNotEqual(StochasticTensor(distrib, samples), t)

    def test_construction(self):
        distrib = Mock(is_reparameterized=True, is_continuous=True)
        samples = tf.constant(12345678., dtype=tf.float32)

        # test basic construction
        t = StochasticTensor(distrib, samples, n_samples=1, group_ndims=2)
        self.assertIs(t.distribution, distrib)
        self.assertTrue(t.is_reparameterized)
        self.assertTrue(t.is_continuous)
        self.assertEqual(t.n_samples, 1)
        self.assertEqual(t.group_ndims, 2)
        self.assertEqual(t.dtype, tf.float32)
        self.assertIsInstance(t.tensor, tf.Tensor)
        with self.test_session():
            self.assertEqual(t.eval(), 12345678.)
            self.assertEqual(t.tensor.eval(), 12345678)

        # test initializing from TensorWrapper
        samples = tf.constant(1.)
        t = StochasticTensor(Mock(is_reparameterized=False),
                             _MyTensorWrapper(samples))
        self.assertIs(t.tensor, samples)

        # test specifying is_reparameterized
        t = StochasticTensor(Mock(is_reparameterized=True), tf.constant(0.),
                             is_reparameterized=False)
        self.assertFalse(t.is_reparameterized)

        # test construction with dynamic group_ndims
        t = StochasticTensor(distrib, samples,
                             group_ndims=tf.constant(2, dtype=tf.int32))
        with self.test_session():
            self.assertEqual(t.group_ndims.eval(), 2)

        # test construction log_prob
        t = StochasticTensor(distrib, tf.constant(0.), log_prob=123.)
        with self.test_session() as sess:
            self.assertEqual(sess.run(t.log_prob()), 123.)

        # test construction with bad dynamic group_ndims
        t = StochasticTensor(distrib, samples,
                             group_ndims=tf.constant(-1, dtype=tf.int32))
        with self.test_session():
            with pytest.raises(Exception,
                               match='group_ndims must be non-negative'):
                _ = t.group_ndims.eval()

        # test construction with dynamic n_samples
        t = StochasticTensor(distrib, samples,
                             n_samples=tf.constant(2, dtype=tf.int32))
        with self.test_session():
            self.assertEqual(t.n_samples.eval(), 2)

        # test construction with bad dynamic n_samples
        t = StochasticTensor(distrib, samples,
                             n_samples=tf.constant(0, dtype=tf.int32))
        with self.test_session():
            with pytest.raises(Exception,
                               match='n_samples must be positive'):
                _ = t.n_samples.eval()

    def test_prob_and_log_prob(self):
        # test default group_ndims
        distrib = Mock(
            is_reparameterized=True,
            log_prob=Mock(return_value=tf.constant(1.)),
            prob=Mock(return_value=tf.constant(2.)),
        )
        t = StochasticTensor(distrib, tf.constant(0.))
        given = t.tensor
        exp_1 = np.exp(1.).astype(np.float32)
        with self.test_session():
            self.assertEqual(t.log_prob().eval(), 1.)
            self.assertEqual(t.log_prob().eval(), 1.)
            np.testing.assert_allclose(t.prob().eval(), exp_1)
            np.testing.assert_allclose(t.prob().eval(), exp_1)
        self.assertEqual(
            distrib.log_prob.call_args_list,
            [((given, 0), {'name': None})]
        )
        self.assertEqual(distrib.prob.call_args_list, [])

        # test group_ndims equal to default
        distrib.log_prob.reset_mock()
        distrib.prob.reset_mock()
        with self.test_session():
            self.assertEqual(t.log_prob(group_ndims=0).eval(), 1.)
            np.testing.assert_allclose(t.prob(group_ndims=0).eval(), exp_1)
        distrib.log_prob.assert_not_called()
        distrib.prob.assert_not_called()

        # test group_ndims different from default
        distrib.log_prob.reset_mock()
        distrib.prob.reset_mock()
        with self.test_session():
            self.assertEqual(t.log_prob(group_ndims=1).eval(), 1.)
            np.testing.assert_allclose(t.prob(group_ndims=2).eval(), exp_1)
        self.assertEqual(
            distrib.log_prob.call_args_list,
            [((given, 1), {'name': None}), ((given, 2),)]
        )
        self.assertEqual(distrib.prob.call_args_list, [])

        # test use dynamic group_ndims
        t = StochasticTensor(distrib, tf.constant(0.),
                             group_ndims=tf.constant(1, dtype=tf.int32))
        given = t.tensor
        distrib.log_prob.reset_mock()
        distrib.prob.reset_mock()
        with self.test_session():
            self.assertEqual(t.log_prob(group_ndims=t.group_ndims).eval(), 1.)
            self.assertEqual(t.log_prob(group_ndims=t.group_ndims).eval(), 1.)
            np.testing.assert_allclose(
                t.prob(group_ndims=t.group_ndims).eval(), exp_1)
            np.testing.assert_allclose(
                t.prob(group_ndims=t.group_ndims).eval(), exp_1)
        self.assertEqual(
            distrib.log_prob.call_args_list,
            [((given, t.group_ndims), {'name': None})]
        )
        self.assertEqual(distrib.prob.call_args_list, [])

    def test_repr(self):
        t = StochasticTensor(
            Mock(is_reparameterized=False),
            Mock(spec=tf.Tensor, __repr__=Mock(return_value='repr_output'))
        )
        self.assertEqual(repr(t), 'StochasticTensor(repr_output)')
