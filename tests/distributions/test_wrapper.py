import pytest
import numpy as np
import tensorflow as tf
import zhusuan.distributions as zd
from mock import Mock

from tfsnippet.distributions import Distribution, as_distribution, Normal
from tfsnippet.distributions.wrapper import ZhuSuanDistribution
from tfsnippet.stochastic import StochasticTensor


class AsDistributionTestCase(tf.test.TestCase):

    def test_distribution(self):
        d = Normal(mean=0., std=1.)
        distrib = as_distribution(d)
        self.assertIs(distrib, d)

    def test_zs_distribution(self):
        normal = zd.Normal(mean=0., std=1.)
        distrib = as_distribution(normal)
        self.assertIsInstance(distrib, Distribution)
        self.assertIsInstance(distrib, ZhuSuanDistribution)
        self.assertIs(distrib._distribution, normal)

    def test_type_error(self):
        with pytest.raises(
                TypeError, match='Type `int` cannot be casted into `tfsnippet.'
                                 'distributions.Distribution`'):
            _ = as_distribution(1)


class ZhuSuanDistributionTestCase(tf.test.TestCase):

    def test_type_error(self):
        with pytest.raises(
                TypeError, match='`distribution` is not an instance of '
                                 '`zhusuan.distributions.Distribution`'):
            _ = ZhuSuanDistribution(1)

    def test_repr(self):
        distrib = ZhuSuanDistribution(Mock(
            spec=zd.Normal,
            wraps=zd.Normal(mean=0., std=1.),
            __repr__=Mock(return_value='repr_output')
        ))
        self.assertEqual(repr(distrib), 'Distribution(repr_output)')

    def test_proxied_props_and_methods(self):
        zs_distrib = zd.OnehotCategorical(tf.zeros([3, 4, 5]), dtype=tf.int64,
                                          group_ndims=1)
        distrib = ZhuSuanDistribution(zs_distrib)
        self.assertEqual(distrib.value_ndims, 1)
        for attr in ['dtype', 'is_continuous', 'is_reparameterized']:
            self.assertEqual(
                getattr(zs_distrib, attr),
                getattr(distrib, attr),
                msg='Attribute `{}` does not equal'.format(attr)
            )
        for meth in ['get_batch_shape']:
            self.assertEqual(
                getattr(zs_distrib, meth)(),
                getattr(distrib, meth)(),
                msg='Output of method `{}` does not equal'.format(meth)
            )
        with self.test_session():
            for attr in ['batch_shape']:
                np.testing.assert_equal(
                    getattr(zs_distrib, attr).eval(),
                    getattr(distrib, attr).eval(),
                    err_msg='Value of attribute `{}` does not equal'.
                            format(attr)
                )
        # for meth in ['get_value_shape', 'get_batch_shape']:
        #     self.assertEqual(
        #         getattr(zs_distrib, meth)(),
        #         getattr(distrib, meth)(),
        #         msg='Output of method `{}` does not equal'.format(meth)
        #     )
        # with self.test_session():
        #     for attr in ['value_shape', 'batch_shape']:
        #         np.testing.assert_equal(
        #             getattr(zs_distrib, attr).eval(),
        #             getattr(distrib, attr).eval(),
        #             err_msg='Value of attribute `{}` does not equal'.
        #                     format(attr)
        #         )

    def test_sample(self):
        # sample re-parameterized samples from a non-reparameterized
        # distribution should cause an error
        with pytest.raises(RuntimeError,
                           match='.* is not re-parameterized'):
            d = ZhuSuanDistribution(
                Mock(
                    spec=zd.Normal,
                    wraps=zd.Normal(mean=0., std=1.),
                    is_reparameterized=False
                )
            )
            self.assertFalse(d.is_reparameterized)
            _ = d.sample(is_reparameterized=True)

        # test sampling with default is_reparameterized = True
        samples = tf.constant(12345678.)
        d = ZhuSuanDistribution(Mock(
            spec=zd.Normal,
            wraps=zd.Normal(mean=0., std=1.),
            is_reparameterized=True,
            sample=Mock(return_value=samples)
        ))
        t = d.sample()
        self.assertIsInstance(t, StochasticTensor)
        self.assertIsNone(t.n_samples)
        self.assertEqual(t.group_ndims, 0)
        self.assertTrue(t.is_reparameterized)
        with self.test_session():
            np.testing.assert_equal(d.sample().eval(), samples.eval())

        # test sampling with default is_reparameterized = False
        self.assertFalse(ZhuSuanDistribution(Mock(
            spec=zd.Normal,
            wraps=zd.Normal(mean=0., std=1.),
            is_reparameterized=False,
            sample=Mock(return_value=samples)
        )).sample().is_reparameterized)

        # test sampling with n_samples
        t = d.sample(n_samples=2)
        self.assertEqual(t.n_samples, 2)

        # test sampling with overrided is_reparameterized attribute
        t = d.sample(is_reparameterized=False)
        self.assertFalse(t.is_reparameterized)

    def test_is_reparameterized(self):
        x = tf.get_variable(shape=(), dtype=tf.float32, name='x',
                            initializer=tf.constant_initializer(2.))

        def test_is_reparemeterized(distrib_flag, sample_flag=None):
            normal = zd.Normal(mean=x, std=1., is_reparameterized=distrib_flag)
            distrib = ZhuSuanDistribution(normal)
            self.assertEqual(distrib.value_ndims, 0)
            samples = distrib.sample(is_reparameterized=sample_flag)
            grads = tf.gradients(samples, x)
            if sample_flag is True or (sample_flag is None and distrib_flag):
                self.assertIsNotNone(grads[0])
            else:
                self.assertIsNone(grads[0])

        with self.test_session() as sess:
            sess.run(tf.variables_initializer([x]))
            test_is_reparemeterized(distrib_flag=True, sample_flag=None)
            test_is_reparemeterized(distrib_flag=True, sample_flag=True)
            test_is_reparemeterized(distrib_flag=True, sample_flag=False)
            test_is_reparemeterized(distrib_flag=False, sample_flag=None)
            test_is_reparemeterized(distrib_flag=False, sample_flag=False)

    def test_prob_and_log_prob(self):
        x = tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4]) / 24.
        normal = zd.Normal(mean=tf.zeros([3, 4]), std=tf.ones([3, 4]))
        normal1 = zd.Normal(mean=tf.zeros([3, 4]), std=tf.ones([3, 4]),
                            group_ndims=1)

        # test with default group_ndims
        distrib = ZhuSuanDistribution(normal)
        with self.test_session():
            self.assertEqual(distrib.log_prob(x).get_shape(),
                             normal.log_prob(x).get_shape())
            self.assertEqual(distrib.prob(x).get_shape(),
                             normal.prob(x).get_shape())
            np.testing.assert_allclose(distrib.log_prob(x).eval(),
                                       normal.log_prob(x).eval())
            np.testing.assert_allclose(distrib.prob(x).eval(),
                                       normal.prob(x).eval())

        # test with static group_ndims
        with self.test_session():
            self.assertEqual(distrib.log_prob(x, group_ndims=1).get_shape(),
                             normal1.log_prob(x).get_shape())
            self.assertEqual(distrib.prob(x, group_ndims=1).get_shape(),
                             normal1.prob(x).get_shape())
            np.testing.assert_allclose(
                distrib.log_prob(x, group_ndims=1).eval(),
                normal1.log_prob(x).eval()
            )
            np.testing.assert_allclose(
                distrib.prob(x, group_ndims=1).eval(),
                normal1.prob(x).eval(),
                rtol=1e-5
            )

        # test with dynamic group_ndims
        group_ndims = tf.constant(1, dtype=tf.int32)
        normal1d = zd.Normal(mean=normal.mean, std=normal.std,
                             group_ndims=group_ndims)
        with self.test_session():
            # Note: Because we added auxiliary asserts to reduce_mean in our
            # log_prob, the following two static shapes will not be equal.
            #
            # self.assertEqual(
            #     distrib.log_prob(x, group_ndims=group_ndims).get_shape(),
            #     normal1d.log_prob(x).get_shape()
            # )
            # self.assertEqual(
            #     distrib.prob(x, group_ndims=group_ndims).get_shape(),
            #     normal1d.prob(x).get_shape()
            # )
            np.testing.assert_allclose(
                distrib.log_prob(x, group_ndims=group_ndims).eval(),
                normal1d.log_prob(x).eval()
            )
            np.testing.assert_allclose(
                distrib.prob(x, group_ndims=group_ndims).eval(),
                normal1d.prob(x).eval(),
                rtol=1e-5
            )

        # test with bad dynamic group_ndims
        group_ndims = tf.constant(-1, dtype=tf.int32)
        with self.test_session():
            with pytest.raises(Exception,
                               match='group_ndims must be non-negative'):
                _ = distrib.log_prob(x, group_ndims=group_ndims).eval()
            with pytest.raises(Exception,
                               match='group_ndims must be non-negative'):
                _ = distrib.prob(x, group_ndims=group_ndims).eval()

        # test override the default group_ndims in ZhuSuan distribution
        distrib = ZhuSuanDistribution(normal1)
        with self.test_session():
            self.assertEqual(distrib.log_prob(x).get_shape(),
                             normal.log_prob(x).get_shape())
            self.assertEqual(distrib.prob(x).get_shape(),
                             normal.prob(x).get_shape())
            np.testing.assert_allclose(distrib.log_prob(x).eval(),
                                       normal.log_prob(x).eval())
            np.testing.assert_allclose(distrib.prob(x).eval(),
                                       normal.prob(x).eval(),
                                       rtol=1e-5)

        # test compute_density
        distrib = ZhuSuanDistribution(normal1)
        t = distrib.sample()
        self.assertIsNone(t._self_log_prob)
        t = distrib.sample(compute_density=False)
        self.assertIsNone(t._self_log_prob)
        t = distrib.sample(compute_density=True)
        self.assertIsNotNone(t._self_log_prob)
