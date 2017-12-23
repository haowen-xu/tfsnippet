import functools
import pytest
import numpy as np
import tensorflow as tf
import zhusuan as zs
from mock import Mock
from tensorflow import keras as K

from tfsnippet.distributions import (Distribution, DistributionFactory, Normal,
                                     Bernoulli)
from tfsnippet.modules import VAE, Module, Sequential, DictMapper
from tfsnippet.utils import (VarScopeObject, instance_reuse,
                             ensure_variables_initialized)

N_X, N_Z, BATCH_SIZE, X_DIMS, Z_DIMS = 5, 7, 11, 3, 2


class Helper(VarScopeObject):

    def __init__(self):
        super(Helper, self).__init__()
        tf.set_random_seed(1234)

        self.x = tf.get_variable('x',
                                 shape=[BATCH_SIZE, X_DIMS],
                                 initializer=tf.random_normal_initializer())
        self.x2 = tf.get_variable('x2',
                                  shape=[N_X, BATCH_SIZE, X_DIMS],
                                  initializer=tf.random_normal_initializer())
        self.x3 = tf.get_variable('x3',
                                  shape=[N_X, N_Z, BATCH_SIZE, X_DIMS],
                                  initializer=tf.random_normal_initializer())
        self.z = tf.get_variable('z',
                                 shape=[BATCH_SIZE, Z_DIMS],
                                 initializer=tf.random_normal_initializer())
        self.z2 = tf.get_variable('z2',
                                  shape=[N_Z, BATCH_SIZE, Z_DIMS],
                                  initializer=tf.random_normal_initializer())

        self.h_for_p_x = Sequential([
            K.layers.Dense(100, activation=tf.nn.relu),
            DictMapper({'mean': K.layers.Dense(X_DIMS),
                        'logstd': K.layers.Dense(X_DIMS)})
        ])
        self.h_for_q_z = Sequential([
            K.layers.Dense(100, activation=tf.nn.relu),
            DictMapper({'mean': K.layers.Dense(Z_DIMS),
                        'logstd': K.layers.Dense(Z_DIMS)})
        ])

        # ensure variables created
        _ = self.h_for_p_x(self.z)
        _ = self.h_for_q_z(self.x)

    def get_xz_variables(self):
        return self.x, self.x2, self.x3, self.z, self.z2

    def vae(self, is_reparameterized=None, z_group_ndims=1, x_group_ndims=1):
        return VAE(
            p_z=Normal(mean=tf.zeros([BATCH_SIZE, Z_DIMS]),
                       std=tf.ones([BATCH_SIZE, Z_DIMS])),
            p_x_given_z=Normal,
            q_z_given_x=Normal,
            h_for_p_x=self.h_for_p_x,
            h_for_q_z=self.h_for_q_z,
            is_reparameterized=is_reparameterized,
            z_group_ndims=z_group_ndims,
            x_group_ndims=x_group_ndims
        )

    @instance_reuse
    def zs_variational(self, x, observed=None, n_z=None,
                       is_reparameterized=None, z_group_ndims=1):
        if is_reparameterized is None:
            is_reparameterized = True
        with zs.BayesianNet(observed) as net:
            z_params = self.h_for_q_z(x)
            z = zs.Normal('z',
                          mean=z_params['mean'],
                          logstd=z_params['logstd'],
                          is_reparameterized=is_reparameterized,
                          n_samples=n_z,
                          group_ndims=z_group_ndims)
        return net

    @instance_reuse
    def zs_model(self, observed=None, n_x=None, n_z=None,
                 is_reparameterized=None, z_group_ndims=1, x_group_ndims=1):
        if is_reparameterized is None:
            is_reparameterized = True
        with zs.BayesianNet(observed) as net:
            z = zs.Normal('z',
                          mean=tf.zeros([BATCH_SIZE, Z_DIMS]),
                          std=tf.ones([BATCH_SIZE, Z_DIMS]),
                          is_reparameterized=is_reparameterized,
                          n_samples=n_z,
                          group_ndims=z_group_ndims)
            x_params = self.h_for_p_x(z)
            x = zs.Normal('x',
                          mean=x_params['mean'],
                          logstd=x_params['logstd'],
                          n_samples=n_x,
                          group_ndims=x_group_ndims)
        return net

    def zs_objective(self, func, observed=None, latent=None, axis=None,
                     n_z=None):
        with tf.variable_scope(None, default_name='zs_objective'):
            return func(
                log_joint=lambda observed: tf.add_n(
                    self.zs_model(observed, n_z=n_z).local_log_prob(['x', 'z'])
                ),
                observed=observed,
                latent=latent,
                axis=axis,
            )


class VAETestCase(tf.test.TestCase):

    def test_construction(self):
        # test basic
        p_z = Mock(spec=Distribution)
        p_x_given_z = Mock(spec=DistributionFactory)
        q_z_given_x = Mock(spec=DistributionFactory)
        h_for_p_x = Mock(spec=Module)
        h_for_q_z = Mock(spec=Module)
        vae = VAE(p_z=p_z, p_x_given_z=p_x_given_z, q_z_given_x=q_z_given_x,
                  h_for_p_x=h_for_p_x, h_for_q_z=h_for_q_z, z_group_ndims=3,
                  x_group_ndims=2, is_reparameterized=False)
        self.assertIs(vae.p_z, p_z)
        self.assertIs(vae.p_x_given_z, p_x_given_z)
        self.assertIs(vae.q_z_given_x, q_z_given_x)
        self.assertIs(vae.h_for_p_x, h_for_p_x)
        self.assertIs(vae.h_for_q_z, h_for_q_z)
        self.assertIs(vae.z_group_ndims, 3)
        self.assertIs(vae.x_group_ndims, 2)
        self.assertFalse(vae.is_reparameterized)

        # test type error for `p_z`
        with pytest.raises(
                TypeError, match='`p_z` must be an instance of `Distribution`'):
            _ = VAE(p_z=123, p_x_given_z=p_x_given_z, q_z_given_x=q_z_given_x,
                    h_for_p_x=h_for_p_x, h_for_q_z=h_for_q_z)

        # test `p_x_given_z` and `q_z_given_x` from :class:`Distribution`
        vae = VAE(p_z=p_z, p_x_given_z=Bernoulli, q_z_given_x=Normal,
                  h_for_p_x=h_for_p_x, h_for_q_z=h_for_q_z)
        self.assertIsInstance(vae.p_x_given_z, DistributionFactory)
        self.assertIs(vae.p_x_given_z.distribution_class, Bernoulli)
        self.assertIsInstance(vae.q_z_given_x, DistributionFactory)
        self.assertIs(vae.q_z_given_x.distribution_class, Normal)

        # test type error for `p_x_given_z`
        with pytest.raises(
                TypeError,
                match='p_x_given_z must be a subclass of `Distribution`, or '
                      'an instance of `DistributionFactory`'):
            _ = VAE(p_z=p_z, p_x_given_z=object, q_z_given_x=q_z_given_x,
                    h_for_p_x=h_for_p_x, h_for_q_z=h_for_q_z)
        with pytest.raises(
                TypeError,
                match='p_x_given_z must be a subclass of `Distribution`, or '
                      'an instance of `DistributionFactory`'):
            _ = VAE(p_z=p_z, p_x_given_z=object(), q_z_given_x=q_z_given_x,
                    h_for_p_x=h_for_p_x, h_for_q_z=h_for_q_z)

        # test type error for `q_z_given_x`
        with pytest.raises(
                TypeError,
                match='q_z_given_x must be a subclass of `Distribution`, or '
                      'an instance of `DistributionFactory`'):
            _ = VAE(p_z=p_z, p_x_given_z=p_x_given_z, q_z_given_x=object,
                    h_for_p_x=h_for_p_x, h_for_q_z=h_for_q_z)
        with pytest.raises(
                TypeError,
                match='q_z_given_x must be a subclass of `Distribution`, or '
                      'an instance of `DistributionFactory`'):
            _ = VAE(p_z=p_z, p_x_given_z=p_x_given_z, q_z_given_x=object(),
                    h_for_p_x=h_for_p_x, h_for_q_z=h_for_q_z)

        # test type error for `h_for_p_x`
        with pytest.raises(
                TypeError, match='`h_for_p_x` must be an instance of `Module`'):
            _ = VAE(p_z=p_z, p_x_given_z=p_x_given_z, q_z_given_x=q_z_given_x,
                    h_for_p_x=object(), h_for_q_z=h_for_q_z)

        # test type error for `h_for_q_z`
        with pytest.raises(
                TypeError, match='`h_for_q_z` must be an instance of `Module`'):
            _ = VAE(p_z=p_z, p_x_given_z=p_x_given_z, q_z_given_x=q_z_given_x,
                    h_for_p_x=h_for_p_x, h_for_q_z=object())

    def test_variational(self):
        helper = Helper()
        x, x2, x3, z, z2 = helper.get_xz_variables()

        with self.test_session() as sess:
            ensure_variables_initialized()

            # test log-prob of one z observation
            vae = helper.vae()
            q_net = vae.variational(x, z=z)
            q_net_zs = helper.zs_variational(x, observed={'z': z})
            np.testing.assert_allclose(q_net['z'].eval(), z.eval())
            np.testing.assert_allclose(*sess.run(
                [q_net.local_log_prob('z'), q_net_zs.local_log_prob('z')]))

            # test log-prob of multiple z observations
            vae = helper.vae()
            q_net = vae.variational(x, z=z2, n_z=N_Z)
            q_net_zs = helper.zs_variational(x, observed={'z': z2}, n_z=N_Z)
            np.testing.assert_allclose(q_net['z'].eval(), z2.eval())
            np.testing.assert_allclose(*sess.run(
                [q_net.local_log_prob('z'), q_net_zs.local_log_prob('z')]))

            # test the shape of z samples and their log-probs
            vae = helper.vae()
            q_net = vae.variational(x, n_z=N_Z)
            q_net_zs = helper.zs_variational(x, observed={'z': q_net['z']})
            np.testing.assert_equal(
                tf.shape(q_net['z']).eval(), [N_Z, BATCH_SIZE, Z_DIMS])
            np.testing.assert_allclose(*sess.run(
                [q_net.local_log_prob('z'), q_net_zs.local_log_prob('z')]))

    def test_model(self):
        helper = Helper()
        x, x2, x3, z, z2 = helper.get_xz_variables()

        with self.test_session() as sess:
            ensure_variables_initialized()

            # test log-prob of one z observation
            vae = helper.vae()
            p_net = vae.model(z=z, x=x)
            p_net_zs = helper.zs_model(observed={'z': z, 'x': x})
            np.testing.assert_allclose(p_net['z'].eval(), z.eval())
            np.testing.assert_allclose(p_net['x'].eval(), x.eval())
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('z'), p_net_zs.local_log_prob('z')]))
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('x'), p_net_zs.local_log_prob('x')]))

            # test log-prob of multiple x & z observations
            vae = helper.vae()
            p_net = vae.model(z=z2, x=x3, n_z=N_Z, n_x=N_X)
            p_net_zs = helper.zs_model(
                observed={'z': z2, 'x': x3}, n_z=N_Z, n_x=N_X)
            np.testing.assert_allclose(p_net['z'].eval(), z2.eval())
            np.testing.assert_allclose(p_net['x'].eval(), x3.eval())
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('z'), p_net_zs.local_log_prob('z')]))
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('x'), p_net_zs.local_log_prob('x')]))

            # test the shape of samples and their log-probs
            vae = helper.vae()
            p_net = vae.model(n_x=N_X, n_z=N_Z)
            p_net_zs = helper.zs_model(
                observed={'z': p_net['z'], 'x': p_net['x']}, n_z=N_Z, n_x=N_X)
            np.testing.assert_equal(
                tf.shape(p_net['z']).eval(), [N_Z, BATCH_SIZE, Z_DIMS])
            np.testing.assert_equal(
                tf.shape(p_net['x']).eval(), [N_X, N_Z, BATCH_SIZE, X_DIMS])
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('z'), p_net_zs.local_log_prob('z')]))
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('x'), p_net_zs.local_log_prob('x')]))

    def test_group_ndims(self):
        helper = Helper()
        x, x2, x3, z, z2 = helper.get_xz_variables()

        with self.test_session() as sess:
            ensure_variables_initialized()
            vae = helper.vae(z_group_ndims=0, x_group_ndims=2)

            q_net = vae.variational(x, z=z)
            q_net_zs = helper.zs_variational(
                x, observed={'z': z}, z_group_ndims=0)
            np.testing.assert_allclose(*sess.run(
                [q_net.local_log_prob('z'), q_net_zs.local_log_prob('z')]))

            p_net = vae.model(z=z, x=x)
            p_net_zs = helper.zs_model(observed={'z': z, 'x': x},
                                       z_group_ndims=0, x_group_ndims=2)
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('z'), p_net_zs.local_log_prob('z')]))
            np.testing.assert_allclose(*sess.run(
                [p_net.local_log_prob('x'), p_net_zs.local_log_prob('x')]))

    def test_chain(self):
        helper = Helper()
        x, x2, x3, z, z2 = helper.get_xz_variables()

        with self.test_session() as sess:
            ensure_variables_initialized()
            vae = helper.vae()

            # test one z sample
            chain = vae.chain(x)
            q_net, p_net = chain.variational, chain.model
            self.assertEqual(chain.latent_names, ('z',))
            np.testing.assert_allclose(*sess.run([q_net['z'], p_net['z']]))
            np.testing.assert_allclose(*sess.run([p_net['x'], x]))
            self.assertIsNone(chain.latent_axis)

            q_net_zs = helper.zs_variational(x, observed={'z': q_net['z']})
            p_net_zs = helper.zs_model(observed={'z': q_net['z'], 'x': x})
            np.testing.assert_allclose(*sess.run(
                [chain.variational.local_log_prob('z'),
                 q_net_zs.local_log_prob('z')]
            ))
            np.testing.assert_allclose(*sess.run([
                chain.log_joint,
                sum(p_net_zs.local_log_prob(['x', 'z']))
            ]))
            zs_elbo = helper.zs_objective(
                zs.variational.elbo,
                observed={'x': x},
                latent={'z': q_net_zs.query('z', outputs=True,
                                            local_log_prob=True)},
                axis=None,
            )
            np.testing.assert_equal(sess.run(tf.shape(zs_elbo)), [BATCH_SIZE])
            np.testing.assert_allclose(*sess.run(
                [chain.vi.lower_bound.elbo(), zs_elbo]))

            # test multiple z sample
            chain = vae.chain(x, n_z=N_Z)
            q_net, p_net = chain.variational, chain.model
            self.assertEqual(chain.latent_names, ('z',))
            np.testing.assert_allclose(*sess.run([q_net['z'], p_net['z']]))
            np.testing.assert_allclose(*sess.run([p_net['x'], x]))

            # TODO: change latent_axis to -2 once ZhuSuan supports dynamic axis
            np.testing.assert_equal(
                tf.convert_to_tensor(chain.latent_axis).eval(), 0)

            q_net_zs = helper.zs_variational(
                x, observed={'z': q_net['z']}, n_z=N_Z)
            p_net_zs = helper.zs_model(
                observed={'z': q_net['z'], 'x': x}, n_z=N_Z)
            np.testing.assert_allclose(*sess.run(
                [chain.variational.local_log_prob('z'),
                 q_net_zs.local_log_prob('z')]
            ))
            np.testing.assert_allclose(*sess.run([
                chain.log_joint,
                sum(p_net_zs.local_log_prob(['x', 'z']))
            ]))
            zs_importance_weighted_objective = helper.zs_objective(
                zs.variational.importance_weighted_objective,
                observed={'x': x},
                latent={'z': q_net_zs.query('z', outputs=True,
                                            local_log_prob=True)},
                axis=0,  # since we do not have n_x, the axis should be 0
                n_z=N_Z,
            )
            np.testing.assert_equal(
                sess.run(tf.shape(zs_importance_weighted_objective)),
                [BATCH_SIZE]
            )
            np.testing.assert_allclose(*sess.run([
                chain.vi.lower_bound.importance_weighted_objective(),
                zs_importance_weighted_objective
            ]))

    def test_training_objective(self):
        class Capture(object):

            def __init__(self, vae):
                self._chain_func = vae.chain
                self._solvers = {}
                self.called_solver = None
                self.chain = None
                vae.chain = self

            def _call_solver(self, solver_name, *args, **kwargs):
                self.called_solver = solver_name
                return self._solvers[solver_name](*args, **kwargs)

            def __call__(self, *args, **kwargs):
                chain = self._chain_func(*args, **kwargs)
                self.chain = chain
                for k in ['iwae', 'vimco', 'sgvb', 'reinforce']:
                    self._solvers[k] = getattr(chain.vi.training, k)
                    setattr(chain.vi.training, k,
                            functools.partial(self._call_solver, solver_name=k))
                return chain

            @property
            def q_net(self):
                return self.chain.variational

        helper = Helper()
        x, x2, x3, z, z2 = helper.get_xz_variables()

        with self.test_session() as sess:
            ensure_variables_initialized()

            # test error for n_z
            vae = helper.vae()
            with pytest.raises(
                    TypeError, match='Cannot choose the variational solver '
                                     'automatically for dynamic `n_z`'):
                vae.get_training_objective(x, n_z=tf.constant(1))

            # test sgvb
            vae = helper.vae()
            capture = Capture(vae)
            loss = vae.get_training_objective(x)
            self.assertEqual(capture.called_solver, 'sgvb')

            zs_obj = helper.zs_objective(
                zs.variational.elbo,
                observed={'x': x},
                latent={'z': capture.q_net.query(['z'])[0]}
            )
            np.testing.assert_allclose(*sess.run([loss, zs_obj.sgvb()]))

            # test sgvb with explicit n_z
            vae = helper.vae()
            capture = Capture(vae)
            loss = vae.get_training_objective(x, n_z=1)
            self.assertEqual(capture.called_solver, 'sgvb')

            zs_obj = helper.zs_objective(
                zs.variational.elbo,
                observed={'x': x},
                latent={'z': capture.q_net.query(['z'])[0]},
                axis=0,
            )
            np.testing.assert_allclose(*sess.run([loss, zs_obj.sgvb()]))

            # test iwae
            vae = helper.vae()
            capture = Capture(vae)
            loss = vae.get_training_objective(x, n_z=N_Z)
            self.assertEqual(capture.called_solver, 'iwae')

            zs_obj = helper.zs_objective(
                zs.variational.importance_weighted_objective,
                observed={'x': x},
                latent={'z': capture.q_net.query(['z'])[0]},
                axis=0,
            )
            np.testing.assert_allclose(*sess.run([loss, zs_obj.sgvb()]))

            # test reinforce
            vae = helper.vae(is_reparameterized=False)
            capture = Capture(vae)
            loss = vae.get_training_objective(x)
            self.assertEqual(capture.called_solver, 'reinforce')

            # TODO: The output of reinforce mismatches on some platform
            #
            # REINFORCE requires additional moving average variable, causing
            # it very hard to ensure two calls should have identical outputs.
            # So we disable such tests for the time being.

            # zs_obj = helper.zs_objective(
            #     zs.variational.elbo,
            #     observed={'x': x},
            #     latent={'z': capture.q_net.query(['z'])[0]}
            # )
            # with tf.variable_scope(None, default_name='reinforce'):
            #     zs_reinforce = zs_obj.reinforce()
            # ensure_variables_initialized()
            # np.testing.assert_allclose(*sess.run([loss, zs_reinforce]))

            # test reinforce with explicit n_z
            vae = helper.vae(is_reparameterized=False)
            capture = Capture(vae)
            loss = vae.get_training_objective(x, n_z=1)
            self.assertEqual(capture.called_solver, 'reinforce')

            # TODO: The output of reinforce mismatches on some platform

            # zs_obj = helper.zs_objective(
            #     zs.variational.elbo,
            #     observed={'x': x},
            #     latent={'z': capture.q_net.query(['z'])[0]},
            #     axis=0,
            # )
            # with tf.variable_scope(None, default_name='reinforce'):
            #     zs_reinforce = zs_obj.reinforce()
            # ensure_variables_initialized()
            # np.testing.assert_allclose(*sess.run([loss, zs_reinforce]))

            # test vimco
            vae = helper.vae(is_reparameterized=False)
            capture = Capture(vae)
            loss = vae.get_training_objective(x, n_z=N_Z)
            self.assertEqual(capture.called_solver, 'vimco')

            zs_obj = helper.zs_objective(
                zs.variational.importance_weighted_objective,
                observed={'x': x},
                latent={'z': capture.q_net.query(['z'])[0]},
                axis=0,
            )
            np.testing.assert_allclose(*sess.run([loss, zs_obj.vimco()]))

    def test_reconstruct(self):
        class Capture(object):

            def __init__(self, vae):
                self._variational_func = vae.variational
                self._model_func = vae.model
                self.q_net = None
                self.p_net = None

                def variational(*args, **kwargs):
                    self.q_net = self._variational_func(*args, **kwargs)
                    return self.q_net

                def model(*args, **kwargs):
                    self.p_net = self._model_func(*args, **kwargs)
                    return self.p_net

                vae.variational = variational
                vae.model = model

        helper = Helper()
        x, x2, x3, z, z2 = helper.get_xz_variables()

        with self.test_session() as sess:
            ensure_variables_initialized()

            # test single sample
            vae = helper.vae()
            capture = Capture(vae)
            x_r = vae.reconstruct(x)
            np.testing.assert_equal(tf.shape(x_r).eval(), [BATCH_SIZE, X_DIMS])
            p_net_zs = helper.zs_model(
                observed={'z': capture.q_net['z'], 'x': x_r})
            np.testing.assert_allclose(*sess.run([x_r, capture.p_net['x']]))
            np.testing.assert_allclose(*sess.run([
                capture.p_net['z'].log_prob(),
                p_net_zs.local_log_prob('z')
            ]))
            np.testing.assert_allclose(*sess.run([
                x_r.log_prob(),
                p_net_zs.local_log_prob('x')
            ]))

            # test multiple samples
            vae = helper.vae()
            capture = Capture(vae)
            x_r = vae.reconstruct(x, n_z=N_Z, n_x=N_X)
            np.testing.assert_equal(
                tf.shape(x_r).eval(), [N_X, N_Z, BATCH_SIZE, X_DIMS])
            p_net_zs = helper.zs_model(
                observed={'z': capture.q_net['z'], 'x': x_r}, n_z=N_Z, n_x=N_X)
            np.testing.assert_allclose(*sess.run([x_r, capture.p_net['x']]))
            np.testing.assert_allclose(*sess.run([
                capture.p_net['z'].log_prob(),
                p_net_zs.local_log_prob('z')
            ]))
            np.testing.assert_allclose(*sess.run([
                x_r.log_prob(),
                p_net_zs.local_log_prob('x')
            ]))

    def test_forward(self):
        helper = Helper()
        x, x2, x3, z, z2 = helper.get_xz_variables()

        with self.test_session() as sess:
            ensure_variables_initialized()

            # test single sample
            vae = helper.vae()
            z_s = vae(x)
            np.testing.assert_equal(tf.shape(z_s).eval(), [BATCH_SIZE, Z_DIMS])
            q_net_zs = helper.zs_variational(x, observed={'z': z_s})
            np.testing.assert_allclose(*sess.run([
                z_s.log_prob(),
                q_net_zs.local_log_prob('z')
            ]))

            # test multiple samples
            vae = helper.vae()
            z_s = vae(x, n_z=N_Z)
            np.testing.assert_equal(
                tf.shape(z_s).eval(), [N_Z, BATCH_SIZE, Z_DIMS])
            q_net_zs = helper.zs_variational(x, observed={'z': z_s})
            np.testing.assert_allclose(*sess.run([
                z_s.log_prob(),
                q_net_zs.local_log_prob('z')
            ]))
