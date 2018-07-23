import pytest
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.bayes import BayesianNet, TransformedDistribution
from tfsnippet.distributions import Normal, Categorical, ExpConcrete
from tfsnippet.stochastic import StochasticTensor


class TransformedDistributionTestCase(tf.test.TestCase):

    def test_basic(self):
        x = Normal(mean=[0., 1.], logstd=0.).sample(10, group_ndims=1)
        log_p = x.log_prob()
        self.assertListEqual([10, 2], x.get_shape().as_list())
        self.assertListEqual([10], log_p.get_shape().as_list())

        transformed = x * 2.
        transformed_log_p = log_p * .5
        d = TransformedDistribution(x, transformed, transformed_log_p,
                                    is_reparameterized=True,
                                    is_continuous=False)
        self.assertIs(x, d.origin)
        self.assertIs(transformed, d.transformed)
        self.assertIs(transformed_log_p, d.transformed_log_p)
        self.assertEquals(x.dtype, d.dtype)
        self.assertTrue(d.is_reparameterized)
        self.assertFalse(d.is_continuous)
        self.assertIs(transformed_log_p, d.log_prob(transformed, group_ndims=1))

        with self.test_session() as sess:
            np.testing.assert_allclose(
                *sess.run([tf.exp(log_p * .5),
                           d.prob(transformed, group_ndims=1)]))

        with_error = lambda: pytest.raises(
            ValueError, match='`given` must be `self.transformed` and '
                              '`group_ndims` must be `self.origin.group_'
                              'ndims`.')
        with with_error():
            _ = d.log_prob(tf.constant(1.), group_ndims=1)
        with with_error():
            _ = d.log_prob(transformed, group_ndims=0)
        with with_error():
            _ = d.prob(tf.constant(1.), group_ndims=1)
        with with_error():
            _ = d.prob(transformed, group_ndims=0)


class BayesianNetTestCase(tf.test.TestCase):

    def test_observed_dict(self):
        # test no observations
        net = BayesianNet()
        self.assertEqual(net.observed, {})
        with pytest.raises(Exception,
                           message='`net.observed` should be read-only'):
            net.observed['x'] = 1

        # test feeding observed with dict
        net = BayesianNet({'x': 1, 'y': 2})
        self.assertEqual(set(net.observed), {'x', 'y'})
        with self.test_session() as sess:
            self.assertListEqual(
                sess.run([net.observed['x'], net.observed['y']]),
                [1, 2]
            )

    def test_add(self):
        x_observed = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        net = BayesianNet({'x': x_observed})
        self.assertNotIn('x', net)
        self.assertNotIn('y', net)

        # add an observed node
        x = net.add('x',
                    Normal(tf.zeros([3, 4]), tf.ones([3, 4])),
                    n_samples=2,
                    group_ndims=1,
                    is_reparameterized=False)
        self.assertIs(net.get('x'), x)
        self.assertIs(net['x'], x)
        self.assertIn('x', net)
        self.assertListEqual(list(net), ['x'])

        self.assertIsInstance(x, StochasticTensor)
        self.assertEqual(x.n_samples, 2)
        self.assertEqual(x.group_ndims, 1)
        self.assertEqual(x.is_reparameterized, False)
        with self.test_session():
            np.testing.assert_allclose(x.eval(), x_observed)
            np.testing.assert_equal(tf.shape(x).eval(), [2, 3, 4])

        # add an unobserved node
        y = net.add('y',
                    Normal(tf.zeros([3, 4]), tf.ones([3, 4])),
                    n_samples=2,
                    group_ndims=1,
                    is_reparameterized=False)
        self.assertIs(net.get('y'), y)
        self.assertIs(net['y'], y)
        self.assertIn('y', net)
        self.assertListEqual(list(net), ['x', 'y'])

        self.assertIsInstance(y, StochasticTensor)
        self.assertEqual(y.n_samples, 2)
        self.assertEqual(y.group_ndims, 1)
        self.assertEqual(y.is_reparameterized, False)
        with self.test_session():
            np.testing.assert_equal(tf.shape(y).eval(), [2, 3, 4])

        # error adding non-string name
        with pytest.raises(TypeError, match='`name` must be a str'):
            _ = net.add(1, Normal(0., 1.))

        # error adding the same variable
        with pytest.raises(
                KeyError,
                match='StochasticTensor with name \'x\' already exists '
                      'in the BayesianNet.  Names must be unique.'):
            _ = net.add('x', Normal(2., 3.))

        # default is_reparameterized of Normal
        z = net.add('z', Normal(0., 1.))
        self.assertTrue(z.is_reparameterized)

    def test_add_transform(self):
        class PatchedNormal(Normal):
            def sample(self, n_samples=None, group_ndims=0,
                       is_reparameterized=None, name=None):
                return StochasticTensor(
                    self,
                    x_samples,
                    n_samples=n_samples,
                    group_ndims=group_ndims,
                    is_reparameterized=is_reparameterized,
                )

        net = BayesianNet({'w': tf.constant(0.)})
        x_samples = tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4])
        normal = PatchedNormal(tf.zeros([3, 4]), tf.ones([3, 4]))

        # test success call
        x = net.add('x', normal, n_samples=2, group_ndims=1,
                    transform=lambda x, log_p: (x * 2., log_p * .5))
        self.assertIsInstance(x.distribution, TransformedDistribution)
        self.assertEquals(1, x.group_ndims)
        self.assertEquals(2, x.n_samples)
        self.assertTrue(x.is_reparameterized)

        with self.test_session() as sess:
            np.testing.assert_allclose(
                *sess.run([x_samples * 2., x]))
            np.testing.assert_allclose(
                *sess.run([normal.log_prob(x_samples, group_ndims=1) * .5,
                           x.log_prob()]))

        # test errors
        I = lambda x, log_p: (x, log_p)
        with pytest.raises(TypeError,
                           match='Cannot add `TransformedDistribution`'):
            _ = net.add('y', x.distribution)
        with pytest.raises(ValueError,
                           match='`transform` can only be applied on '
                                 'continuous, re-parameterized variables'):
            _ = net.add('y', Categorical([0.], dtype=tf.int32), transform=I)
        with pytest.raises(ValueError,
                           match='`transform` can only be applied on '
                                 'continuous, re-parameterized variables'):
            _ = net.add('y', ExpConcrete(.5, [0.], is_reparameterized=False),
                        transform=I)
        with pytest.raises(ValueError,
                           match='`observed` variable cannot be transformed.'):
            _ = net.add('w', Normal(mean=0., std=0.), transform=I)
        with pytest.raises(ValueError,
                           match='The transformed samples must be continuous'):
            T = lambda x, log_p: (tf.cast(x, dtype=tf.int32), log_p)
            _ = net.add('y', normal, transform=T)

    def test_outputs(self):
        x_observed = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        net = BayesianNet({'x': x_observed})
        normal = Normal(tf.zeros([3, 4]), tf.ones([3, 4]))
        x = net.add('x', normal)
        y = net.add('y', normal)

        # test single query
        x_out = net.output('x')
        self.assertIs(x_out, x.tensor)
        self.assertIsInstance(x_out, tf.Tensor)
        with self.test_session():
            np.testing.assert_equal(x_out.eval(), x_observed)

        # test multiple query
        x_out, y_out = net.outputs(iter(['x', 'y']))
        self.assertIs(x_out, x.tensor)
        self.assertIs(y_out, y.tensor)
        self.assertIsInstance(x_out, tf.Tensor)
        self.assertIsInstance(y_out, tf.Tensor)
        with self.test_session():
            np.testing.assert_equal(x_out.eval(), x_observed)

    def test_local_log_prob(self):
        x_observed = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        net = BayesianNet({'x': x_observed})
        normal = Normal(tf.zeros([3, 4]), tf.ones([3, 4]))
        x = net.add('x', normal)
        y = net.add('y', normal)

        # test single query
        x_log_prob = net.local_log_prob('x')
        self.assertIsInstance(x_log_prob, tf.Tensor)
        with self.test_session():
            np.testing.assert_allclose(
                x_log_prob.eval(), normal.log_prob(x_observed).eval())

        # test multiple query
        x_log_prob, y_log_prob = net.local_log_probs(iter(['x', 'y']))
        self.assertIsInstance(x_log_prob, tf.Tensor)
        self.assertIsInstance(y_log_prob, tf.Tensor)
        with self.test_session() as sess:
            np.testing.assert_allclose(
                x_log_prob.eval(), normal.log_prob(x_observed).eval())
            x_log_prob_val, x_log_prob_res, y_log_prob_val, y_log_prob_res = \
                sess.run([
                    x_log_prob, normal.log_prob(x.tensor),
                    y_log_prob, normal.log_prob(y.tensor),
                ])
            np.testing.assert_allclose(x_log_prob_val, x_log_prob_res)
            np.testing.assert_allclose(y_log_prob_val, y_log_prob_res)

    def test_query(self):
        x_observed = np.arange(24, dtype=np.float32).reshape([2, 3, 4])
        net = BayesianNet({'x': x_observed})
        normal = Normal(tf.zeros([3, 4]), tf.ones([3, 4]))
        x = net.add('x', normal)
        y = net.add('y', normal)

        [(x_out, x_log_prob), (y_out, y_log_prob)] = net.query(iter(['x', 'y']))
        for o in [x_out, x_log_prob, y_out, y_log_prob]:
            self.assertIsInstance(o, tf.Tensor)
        self.assertIs(x_out, x.tensor)
        self.assertIs(y_out, y.tensor)
        with self.test_session() as sess:
            np.testing.assert_allclose(
                x_log_prob.eval(), normal.log_prob(x_observed).eval())
            x_log_prob_val, x_log_prob_res, y_log_prob_val, y_log_prob_res = \
                sess.run([
                    x_log_prob, normal.log_prob(x.tensor),
                    y_log_prob, normal.log_prob(y.tensor),
                ])
            np.testing.assert_allclose(x_log_prob_val, x_log_prob_res)
            np.testing.assert_allclose(y_log_prob_val, y_log_prob_res)

    def test_validate_names(self):
        net = BayesianNet({'x': [2., 3., 4.]})
        x = net.add('x', Normal(0., 1.))
        y = net.add('y', Normal(0., 1.))

        for meth in ['output', 'local_log_prob']:
            with pytest.raises(TypeError, match='`names` is not a list of str'):
                _ = getattr(net, meth)(1)
            with pytest.raises(KeyError, match='StochasticTensor with name '
                                               '\'z\' does not exist'):
                _ = getattr(net, meth)('z')
        for meth in ['outputs', 'local_log_probs', 'query']:
            with pytest.raises(TypeError, match='`names` is not a list of str'):
                _ = getattr(net, meth)([1, 2])
            with pytest.raises(KeyError, match='StochasticTensor with name '
                                               '\'z\' does not exist'):
                _ = getattr(net, meth)(['x', 'y', 'z'])

    def test_variational_chain(self):
        q_net = BayesianNet({'x': [1.]})
        q_net.add('z', Normal(q_net.observed['x'], 1.))
        q_net.add('y', Normal(q_net.observed['x'] * 2, 2.))

        def model_builder(observed):
            model = BayesianNet(observed)
            z = model.add('z', Normal([0.], [1.]))
            y = model.add('y', Normal([0.], [2.]))
            x = model.add('x', Normal(z + y, [1.]))
            return model

        model_builder = Mock(wraps=model_builder)

        # test chain with default parameters
        chain = q_net.variational_chain(model_builder)
        self.assertEqual(
            model_builder.call_args,
            (({'y': q_net['y'].tensor, 'z': q_net['z'].tensor},),)
        )
        self.assertEqual(chain.latent_names, ('z', 'y'))
        self.assertIsNone(chain.latent_axis)

        # test chain with latent_names
        chain = q_net.variational_chain(model_builder, latent_names=['y'])
        self.assertEqual(
            model_builder.call_args,
            (({'y': q_net['y'].tensor},),)
        )
        self.assertEqual(chain.latent_names, ('y',))

        # test chain with latent_axis
        chain = q_net.variational_chain(model_builder, latent_axis=-1)
        self.assertEqual(chain.latent_axis, -1)

        # test chain with observed
        chain = q_net.variational_chain(model_builder, observed=q_net.observed)
        self.assertEqual(
            model_builder.call_args,
            (({'x': q_net.observed['x'], 'y': q_net['y'].tensor,
               'z': q_net['z'].tensor},),)
        )
        self.assertEqual(chain.latent_names, ('z', 'y'))

        # test model_builder with log_joint
        def model_builder_1(observed):
            return model_builder(observed), fake_log_joint

        fake_log_joint = tf.constant(0.)
        chain = q_net.variational_chain(model_builder_1)

        with self.test_session():
            np.testing.assert_equal(
                chain.log_joint.eval(), fake_log_joint.eval())
