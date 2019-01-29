import functools
from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf
import zhusuan as zs

from tfsnippet.ops import add_n_broadcast
from tfsnippet.utils import ensure_variables_initialized
from tfsnippet.variational import *


class VariationalInferenceTestCase(tf.test.TestCase):

    def test_construction(self):
        vi = VariationalInference(
            tf.constant(1.), [tf.constant(2.), tf.constant(3.)])
        self.assertIsNone(vi.axis)
        self.assertIsInstance(vi.training, VariationalTrainingObjectives)
        self.assertIsInstance(vi.lower_bound, VariationalLowerBounds)
        self.assertIsInstance(vi.evaluation, VariationalEvaluation)

        with self.test_session():
            np.testing.assert_equal(vi.log_joint.eval(), 1.)
            np.testing.assert_equal([t.eval() for t in vi.latent_log_probs],
                                    [2., 3.])

    def test_errors(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(tf.constant(0.), [tf.constant(0.)],
                                  axis=None)
        with pytest.raises(
                ValueError, match='monte carlo objective requires '
                                  'multi-samples'):
            _ = vi.lower_bound.importance_weighted_objective()
        with pytest.raises(
                ValueError, match='iwae training objective requires '
                                  'multi-samples'):
            _ = vi.training.iwae()
        with pytest.raises(
                ValueError, match='vimco training objective requires '
                                  'multi-samples'):
            _ = vi.training.vimco()
        with pytest.raises(
                ValueError, match='reweighted wake-sleep training objective '
                                  'requires multi-samples'):
            _ = vi.training.rws_wake()

    def test_elbo(self):
        with self.test_session() as sess:
            log_p = tf.random_normal(shape=[5, 7])
            log_q1 = tf.random_normal(shape=[1, 3, 5, 7])
            log_q2 = tf.random_normal(shape=[4, 1, 5, 7])

            # test without sampling axis
            vi = VariationalInference(log_p, [log_q1, log_q2])
            output = vi.lower_bound.elbo()
            answer = elbo_objective(log_p, log_q1 + log_q2)
            np.testing.assert_allclose(*sess.run([output, answer]))

            # test with sampling axis
            vi = VariationalInference(log_p, [log_q1, log_q2], axis=[0, 1])
            output = vi.lower_bound.elbo()
            answer = elbo_objective(
                log_p, log_q1 + log_q2, axis=[0, 1])
            np.testing.assert_allclose(*sess.run([output, answer]))

    def test_monte_carlo_objective(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(tf.constant(0.), [tf.constant(0.)],
                                  axis=None)
        with pytest.raises(
                ValueError, match='monte carlo objective '
                                  'requires multi-samples'):
            _ = vi.lower_bound.monte_carlo_objective()

        # test with sampling axis
        with self.test_session() as sess:
            log_p = tf.random_normal(shape=[5, 7])
            log_q1 = tf.random_normal(shape=[1, 3, 5, 7])
            log_q2 = tf.random_normal(shape=[4, 1, 5, 7])
            vi = VariationalInference(log_p, [log_q1, log_q2], axis=[0, 1])
            output = vi.lower_bound.monte_carlo_objective()
            answer = monte_carlo_objective(
                log_p, log_q1 + log_q2, axis=[0, 1])
            np.testing.assert_allclose(*sess.run([output, answer]))

    def test_sgvb(self):
        with self.test_session() as sess:
            log_p = tf.random_normal(shape=[5, 7])
            log_q1 = tf.random_normal(shape=[1, 3, 5, 7])
            log_q2 = tf.random_normal(shape=[4, 1, 5, 7])

            # test without sampling axis
            vi = VariationalInference(log_p, [log_q1, log_q2])
            output = vi.training.sgvb()
            answer = -sgvb_estimator(log_p - (log_q1 + log_q2))
            np.testing.assert_allclose(*sess.run([output, answer]))

            # test with sampling axis
            vi = VariationalInference(log_p, [log_q1, log_q2], axis=[0, 1])
            output = vi.training.sgvb()
            answer = -sgvb_estimator(
                log_p - (log_q1 + log_q2), axis=[0, 1])
            np.testing.assert_allclose(*sess.run([output, answer]))

    def test_nvil(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        with self.test_session() as sess:
            log_p = tf.random_normal(shape=[5, 7])
            log_q1 = tf.random_normal(shape=[1, 3, 5, 7])
            log_q2 = tf.random_normal(shape=[4, 1, 5, 7])
            baseline = tf.random_normal(shape=[5, 7])

            # test without sampling axis
            vi = VariationalInference(log_p, [log_q1, log_q2])
            output = vi.training.nvil(baseline=None,
                                      center_by_moving_average=True)
            a, b = nvil_estimator(
                values=log_p - (log_q1 + log_q2),
                latent_log_joint=log_q1 + log_q2,
                baseline=None,
                center_by_moving_average=True,
                decay=0.8,
                alt_values=log_p
            )
            answer = - a
            ensure_variables_initialized()
            assert_allclose(*sess.run([output, answer]))

            # test with sampling axis
            vi = VariationalInference(log_p, [log_q1, log_q2], axis=[0, 1])
            output = vi.training.nvil(baseline=baseline,
                                      center_by_moving_average=False)
            a, b = nvil_estimator(
                values=log_p - (log_q1 + log_q2),
                latent_log_joint=log_q1 + log_q2,
                axis=[0, 1],
                baseline=baseline,
                center_by_moving_average=False,
                decay=0.8,
                alt_values=log_p
            )
            answer = b - a
            ensure_variables_initialized()
            assert_allclose(*sess.run([output, answer]))

    def test_iwae(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(tf.constant(0.), [tf.constant(0.)],
                                  axis=None)
        with pytest.raises(
                ValueError, match='iwae training objective '
                                  'requires multi-samples'):
            _ = vi.training.iwae()

        with self.test_session() as sess:
            log_p = tf.random_normal(shape=[5, 7])
            log_q1 = tf.random_normal(shape=[1, 3, 5, 7])
            log_q2 = tf.random_normal(shape=[4, 1, 5, 7])

            vi = VariationalInference(log_p, [log_q1, log_q2], axis=[0, 1])
            output = vi.training.iwae()
            answer = -iwae_estimator(
                log_p - (log_q1 + log_q2), axis=[0, 1])
            np.testing.assert_allclose(*sess.run([output, answer]))

    def test_is_loglikelihood(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(tf.constant(0.), [tf.constant(0.)],
                                  axis=None)
        with pytest.raises(
                ValueError, match='importance sampling log-likelihood '
                                  'requires multi-samples'):
            _ = vi.evaluation.importance_sampling_log_likelihood()

        # test with sampling axis
        with self.test_session() as sess:
            log_p = tf.random_normal(shape=[5, 7])
            log_q1 = tf.random_normal(shape=[1, 3, 5, 7])
            log_q2 = tf.random_normal(shape=[4, 1, 5, 7])
            vi = VariationalInference(log_p, [log_q1, log_q2], axis=[0, 1])
            output = vi.evaluation.importance_sampling_log_likelihood()
            answer = importance_sampling_log_likelihood(
                log_p, log_q1 + log_q2, axis=[0, 1])
            np.testing.assert_allclose(*sess.run([output, answer]))


class VariationalInferenceZhuSuanTestCase(tf.test.TestCase):

    def prepare_model(self, zs_func, axis, n_z):
        PreparedModel = namedtuple(
            'PreparedModel',
            ['model_func', 'log_joint_func', 'q_net', 'zs_obj', 'log_joint',
             'vi']
        )

        x = tf.constant([1., 2., 3.])
        with zs.BayesianNet() as q_net:
            z_posterior = zs.Normal('z', mean=x, std=tf.ones([3]),
                                    n_samples=n_z)

        def model_func(observed):
            with zs.BayesianNet(observed) as net:
                z = zs.Normal('z', mean=tf.zeros([3]), std=tf.ones([3]),
                              n_samples=n_z)
                x = zs.Normal('x', mean=z, std=tf.ones([3]))
            return net

        def log_joint_func(observed):
            net = model_func(observed)
            return add_n_broadcast(net.local_log_prob(['z', 'x']))

        # derive :class:`zhusuan.variational.VariationalObjective`
        # by ZhuSuan utilities
        latent = {'z': q_net.query(['z'], outputs=True,
                                   local_log_prob=True)[0]}
        zs_obj = zs_func(log_joint_func, observed={'x': x}, latent=latent,
                         axis=axis)

        # derive :class:`zhusuan.variational.VariationalObjective`
        # by :class:`VariationalInference`
        log_joint = log_joint_func({'z': q_net.outputs('z'), 'x': x})
        vi = VariationalInference(log_joint, [q_net.local_log_prob('z')],
                                  axis=axis)

        return PreparedModel(model_func, log_joint_func, q_net, zs_obj,
                             log_joint, vi)

    def test_elbo(self):
        # test no sampling axis
        with self.test_session() as sess:
            prepared = self.prepare_model(
                zs.variational.elbo, axis=None, n_z=None)
            vi = prepared.vi
            zs_obj = prepared.zs_obj

            # test :meth:`VariationalInference.zs_objective`
            vi_obj = prepared.vi.zs_objective(zs.variational.elbo)
            self.assertIsInstance(
                vi_obj, zs.variational.EvidenceLowerBoundObjective)
            np.testing.assert_allclose(*sess.run([zs_obj, vi_obj]))

            # test :meth:`VariationalInference.zs_elbo`
            vi_obj = prepared.vi.zs_elbo()
            self.assertIsInstance(
                vi_obj, zs.variational.EvidenceLowerBoundObjective)
            np.testing.assert_allclose(*sess.run([zs_obj, vi_obj]))

        # test with sampling axis
        with self.test_session() as sess:
            prepared = self.prepare_model(
                zs.variational.elbo, axis=0, n_z=7)
            vi = prepared.vi
            zs_obj = prepared.zs_obj

            # test :meth:`VariationalInference.zs_objective`
            vi_obj = prepared.vi.zs_objective(zs.variational.elbo)
            self.assertIsInstance(
                vi_obj, zs.variational.EvidenceLowerBoundObjective)
            np.testing.assert_allclose(*sess.run([zs_obj, vi_obj]))

            # test :meth:`VariationalInference.zs_elbo`
            vi_obj = prepared.vi.zs_elbo()
            self.assertIsInstance(
                vi_obj, zs.variational.EvidenceLowerBoundObjective)
            np.testing.assert_allclose(*sess.run([zs_obj, vi_obj]))

    def test_importance_weighted_objective(self):
        with self.test_session() as sess:
            prepared = self.prepare_model(
                zs.variational.importance_weighted_objective, axis=0, n_z=7)
            vi = prepared.vi
            zs_obj = prepared.zs_obj

            # test :meth:`VariationalInference.zs_objective`
            vi_obj = prepared.vi.zs_objective(
                zs.variational.importance_weighted_objective)
            self.assertIsInstance(
                vi_obj, zs.variational.ImportanceWeightedObjective)
            np.testing.assert_allclose(*sess.run([zs_obj, vi_obj]))

            # test :meth:`VariationalInference.zs_importance_weighted_objective`
            vi_obj = prepared.vi.zs_importance_weighted_objective()
            self.assertIsInstance(
                vi_obj, zs.variational.ImportanceWeightedObjective)
            np.testing.assert_allclose(*sess.run([zs_obj, vi_obj]))

            # test :meth:`VariationalTrainingObjectives.vimco`
            np.testing.assert_allclose(
                *sess.run([zs_obj.vimco(), vi.training.vimco()]))

    def test_klpq(self):
        with self.test_session() as sess:
            prepared = self.prepare_model(zs.variational.klpq, axis=0, n_z=7)
            vi = prepared.vi
            zs_obj = prepared.zs_obj

            # test :meth:`VariationalInference.zs_objective`
            vi_obj = prepared.vi.zs_objective(zs.variational.klpq)
            self.assertIsInstance(vi_obj, zs.variational.InclusiveKLObjective)

            # test :meth:`VariationalInference.zs_klpq`
            vi_obj = prepared.vi.zs_klpq()
            self.assertIsInstance(vi_obj, zs.variational.InclusiveKLObjective)

            # test :meth:`VariationalTrainingObjectives.rws_wake`
            np.testing.assert_allclose(
                *sess.run([zs_obj.rws(), vi.training.rws_wake()]))
