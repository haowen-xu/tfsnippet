import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.distributions import *
from tfsnippet.utils import scoped_set_config, settings


class NormalTestCase(tf.test.TestCase):

    def test_props(self):
        mean = np.asarray([1., 2., -3.], dtype=np.float32)
        std = np.asarray([1.1, 2.2, 3.3], dtype=np.float32)
        logstd = np.log(std)

        with self.test_session():
            # test construction with std
            normal = Normal(mean=mean, std=std)
            self.assertEqual(normal.value_ndims, 0)
            np.testing.assert_allclose(normal.mean.eval(), mean)
            np.testing.assert_allclose(normal.std.eval(), std)
            np.testing.assert_allclose(normal.logstd.eval(), logstd)

            # test construction with logstd
            normal = Normal(mean=mean, logstd=logstd)
            self.assertEqual(normal.value_ndims, 0)
            np.testing.assert_allclose(normal.mean.eval(), mean)
            np.testing.assert_allclose(normal.std.eval(), std)
            np.testing.assert_allclose(normal.logstd.eval(), logstd)

    def test_check_numerics(self):
        with scoped_set_config(settings, check_numerics=True):
            normal = Normal(mean=0., std=-1.)
            with self.test_session():
                with pytest.raises(
                        Exception, match=r'log\(std\) : Tensor had NaN values'):
                    _ = normal.logstd.eval()


class BernoulliTestCase(tf.test.TestCase):

    def test_props(self):
        logits = np.asarray([0., 1., -2.], dtype=np.float32)
        bernoulli = Bernoulli(logits=logits)
        self.assertEqual(bernoulli.value_ndims, 0)
        with self.test_session():
            np.testing.assert_allclose(bernoulli.logits.eval(), logits)

    def test_dtype(self):
        bernoulli = Bernoulli(logits=0.)
        self.assertEqual(bernoulli.dtype, tf.int32)
        samples = bernoulli.sample()
        self.assertEqual(samples.dtype, tf.int32)

        bernoulli = Bernoulli(logits=0., dtype=tf.int64)
        self.assertEqual(bernoulli.dtype, tf.int64)
        samples = bernoulli.sample()
        self.assertEqual(samples.dtype, tf.int64)


class CategoricalTestCase(tf.test.TestCase):

    def test_props(self):
        logits = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        categorical = Categorical(logits=tf.constant(logits))
        self.assertEqual(categorical.value_ndims, 0)
        self.assertEqual(categorical.n_categories, 4)
        with self.test_session():
            np.testing.assert_allclose(categorical.logits.eval(), logits)

    def test_dtype(self):
        categorical = Categorical(logits=tf.constant([0., 1.]))
        self.assertEqual(categorical.dtype, tf.int32)
        samples = categorical.sample()
        self.assertEqual(samples.dtype, tf.int32)

        categorical = Categorical(logits=tf.constant([0., 1.]), dtype=tf.int64)
        self.assertEqual(categorical.dtype, tf.int64)
        samples = categorical.sample()
        self.assertEqual(samples.dtype, tf.int64)


class UniformTestCase(tf.test.TestCase):

    def test_props(self):
        uniform = Uniform(minval=-1., maxval=2.)
        self.assertEqual(uniform.value_ndims, 0)
        with self.test_session():
            self.assertEqual(uniform.minval.eval(), -1.)
            self.assertEqual(uniform.maxval.eval(), 2.)

    def test_check_numerics(self):
        with scoped_set_config(settings, check_numerics=True):
            uniform = Uniform(minval=-1e100, maxval=1e100)
            with self.test_session():
                with pytest.raises(
                        Exception, match=r'log_p : Tensor had Inf values'):
                    _ = uniform.log_prob(0.).eval()
