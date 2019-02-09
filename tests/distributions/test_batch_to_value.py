import numpy as np
import pytest
import tensorflow as tf

from tfsnippet import Normal, BatchToValueDistribution, OnehotCategorical
from tfsnippet.utils import get_static_shape, set_random_seed


class BatchToValueDistributionTestCase(tf.test.TestCase):

    def test_ndims_equals_zero_and_negative(self):
        normal = Normal(mean=tf.zeros([3, 4]), logstd=0.)

        self.assertIs(normal.batch_ndims_to_value(0), normal)
        self.assertIs(normal.expand_value_ndims(0), normal)

        with pytest.raises(ValueError,
                           match='`ndims` must be non-negative integers'):
            _ = normal.batch_ndims_to_value(-1)
        with pytest.raises(ValueError,
                           match='`ndims` must be non-negative integers'):
            _ = normal.expand_value_ndims(-1)

    def test_ndims_exceed_limit(self):
        normal = Normal(mean=tf.zeros([3, 4]), logstd=0.)

        with pytest.raises(ValueError, match='`distribution.batch_shape.ndims` '
                                             'is less then `ndims`'):
            _ = normal.expand_value_ndims(3)

    def test_transform_on_transformed(self):
        with self.test_session() as sess:
            normal = Normal(mean=tf.zeros([3, 4, 5]), logstd=0.)
            self.assertEqual(normal.value_ndims, 0)
            self.assertEqual(normal.get_batch_shape().as_list(), [3, 4, 5])
            self.assertEqual(list(sess.run(normal.batch_shape)), [3, 4, 5])

            distrib = normal.batch_ndims_to_value(0)
            self.assertIs(distrib, normal)

            distrib = normal.batch_ndims_to_value(1)
            self.assertIsInstance(distrib, BatchToValueDistribution)
            self.assertEqual(distrib.value_ndims, 1)
            self.assertEqual(distrib.get_batch_shape().as_list(), [3, 4])
            self.assertEqual(list(sess.run(distrib.batch_shape)), [3, 4])
            self.assertIs(distrib.base_distribution, normal)

            distrib2 = distrib.expand_value_ndims(1)
            self.assertIsInstance(distrib2, BatchToValueDistribution)
            self.assertEqual(distrib2.value_ndims, 2)
            self.assertEqual(distrib2.get_batch_shape().as_list(), [3])
            self.assertEqual(list(sess.run(distrib2.batch_shape)), [3])
            self.assertIs(distrib.base_distribution, normal)

            distrib2 = distrib.expand_value_ndims(0)
            self.assertIs(distrib2, distrib)
            self.assertEqual(distrib2.value_ndims, 1)
            self.assertEqual(distrib.value_ndims, 1)
            self.assertEqual(distrib2.get_batch_shape().as_list(), [3, 4])
            self.assertEqual(list(sess.run(distrib2.batch_shape)), [3, 4])
            self.assertIs(distrib.base_distribution, normal)

    def test_with_normal(self):
        mean = np.random.normal(size=[4, 5]).astype(np.float64)
        logstd = np.random.normal(size=mean.shape).astype(np.float64)
        x = np.random.normal(size=[3, 4, 5])

        with self.test_session() as sess:
            normal = Normal(mean=mean, logstd=logstd)
            distrib = normal.batch_ndims_to_value(1)

            self.assertIsInstance(distrib, BatchToValueDistribution)
            self.assertEqual(distrib.value_ndims, 1)
            self.assertEqual(distrib.get_batch_shape().as_list(), [4])
            self.assertEqual(list(sess.run(distrib.batch_shape)), [4])
            self.assertEqual(distrib.dtype, tf.float64)
            self.assertTrue(distrib.is_continuous)
            self.assertTrue(distrib.is_reparameterized)
            self.assertIs(distrib.base_distribution, normal)

            log_prob = distrib.log_prob(x)
            log_prob2 = distrib.log_prob(x, group_ndims=1)
            self.assertEqual(get_static_shape(log_prob), (3, 4))
            self.assertEqual(get_static_shape(log_prob2), (3,))
            np.testing.assert_allclose(
                *sess.run([log_prob, normal.log_prob(x, group_ndims=1)]))
            np.testing.assert_allclose(
                *sess.run([log_prob2, normal.log_prob(x, group_ndims=2)]))

            prob = distrib.prob(x)
            prob2 = distrib.prob(x, group_ndims=1)
            self.assertEqual(get_static_shape(prob), (3, 4))
            self.assertEqual(get_static_shape(prob2), (3,))
            np.testing.assert_allclose(
                *sess.run([prob, normal.prob(x, group_ndims=1)]))
            np.testing.assert_allclose(
                *sess.run([prob2, normal.prob(x, group_ndims=2)]))

            sample = distrib.sample(3, compute_density=False)
            sample2 = distrib.sample(3, compute_density=True, group_ndims=1)
            log_prob = sample.log_prob()
            log_prob2 = sample2.log_prob()
            self.assertEqual(get_static_shape(log_prob), (3, 4))
            self.assertEqual(get_static_shape(log_prob2), (3,))
            np.testing.assert_allclose(
                *sess.run([log_prob, normal.log_prob(sample, group_ndims=1)]))
            np.testing.assert_allclose(
                *sess.run([log_prob2, normal.log_prob(sample2, group_ndims=2)]))

    def test_with_one_hot_categorical(self):
        set_random_seed(1234)
        logits = np.random.normal(size=[4, 5, 7]).astype(np.float64)

        with self.test_session() as sess:
            cat = OnehotCategorical(logits=logits, dtype=tf.int32)
            x = sess.run(cat.sample(3))
            distrib = cat.batch_ndims_to_value(1)

            self.assertIsInstance(distrib, BatchToValueDistribution)
            self.assertEqual(distrib.value_ndims, 2)
            self.assertEqual(distrib.get_batch_shape().as_list(), [4])
            self.assertEqual(list(sess.run(distrib.batch_shape)), [4])
            self.assertEqual(distrib.dtype, tf.int32)
            self.assertFalse(distrib.is_continuous)
            self.assertFalse(distrib.is_reparameterized)
            self.assertIs(distrib.base_distribution, cat)

            log_prob = distrib.log_prob(x)
            log_prob2 = distrib.log_prob(x, group_ndims=1)
            self.assertEqual(get_static_shape(log_prob), (3, 4))
            self.assertEqual(get_static_shape(log_prob2), (3,))
            np.testing.assert_allclose(
                *sess.run([log_prob, cat.log_prob(x, group_ndims=1)]))
            np.testing.assert_allclose(
                *sess.run([log_prob2, cat.log_prob(x, group_ndims=2)]))

            prob = distrib.prob(x)
            prob2 = distrib.prob(x, group_ndims=1)
            self.assertEqual(get_static_shape(prob), (3, 4))
            self.assertEqual(get_static_shape(prob2), (3,))
            np.testing.assert_allclose(
                *sess.run([prob, cat.prob(x, group_ndims=1)]))
            np.testing.assert_allclose(
                *sess.run([prob2, cat.prob(x, group_ndims=2)]))

            sample = distrib.sample(3, compute_density=False)
            sample2 = distrib.sample(3, compute_density=True, group_ndims=1)
            log_prob = sample.log_prob()
            log_prob2 = sample2.log_prob()
            self.assertEqual(get_static_shape(log_prob), (3, 4))
            self.assertEqual(get_static_shape(log_prob2), (3,))
            np.testing.assert_allclose(
                *sess.run([log_prob, cat.log_prob(sample, group_ndims=1)]))
            np.testing.assert_allclose(
                *sess.run([log_prob2, cat.log_prob(sample2, group_ndims=2)]))
