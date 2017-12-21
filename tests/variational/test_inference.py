from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf
import zhusuan as zs

from tfsnippet.utils import ensure_variables_initialized
from tfsnippet.variational import VariationalInference
from tfsnippet.variational.inference import (VariationalTrainingObjectives,
                                             VariationalLowerBounds)


class VariationalInferenceTestCase(tf.test.TestCase):

    def test_construction(self):
        vi = VariationalInference(
            tf.constant(1.), [tf.constant(2.), tf.constant(3.)])
        self.assertIsNone(vi.axis)
        self.assertIsInstance(vi.training, VariationalTrainingObjectives)
        self.assertIsInstance(vi.lower_bound, VariationalLowerBounds)

        with self.test_session():
            np.testing.assert_equal(vi.log_joint.eval(), 1.)
            np.testing.assert_equal([t.eval() for t in vi.latent_log_probs],
                                    [2., 3.])

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
            return tf.add_n(net.local_log_prob(['z', 'x']))

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

            # test :meth:`VariationalLowerBounds.elbo`
            np.testing.assert_allclose(
                *sess.run([zs_obj, vi.lower_bound.elbo()]))

            # test :meth:`VariationalTrainingObjectives.sgvb`
            np.testing.assert_allclose(
                *sess.run([zs_obj.sgvb(), vi.training.sgvb()]))

            # test :meth:`VariationalTrainingObjectives.reinforce`
            # TODO: The output of reinforce mismatches on some platform
            #       if variance_reduction == True
            # REINFORCE requires additional moving average variable, causing
            # it very hard to ensure two calls should have identical outputs.
            # So we disable such tests for the time being.
            with tf.variable_scope(None, default_name='reinforce'):
                # reinforce requires extra variables, but ZhuSuan does not
                # obtain a dedicated variable scope.  so we open one here.
                zs_reinforce = zs_obj.reinforce(variance_reduction=False)
            vi_reinforce = vi.training.reinforce(variance_reduction=False)
            ensure_variables_initialized()
            np.testing.assert_allclose(*sess.run([zs_reinforce, vi_reinforce]))

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

            # test :meth:`VariationalLowerBounds.elbo`
            np.testing.assert_allclose(
                *sess.run([zs_obj, vi.lower_bound.elbo()]))

            # test :meth:`VariationalTrainingObjectives.sgvb`
            np.testing.assert_allclose(
                *sess.run([zs_obj.sgvb(), vi.training.sgvb()]))

            # test :meth:`VariationalTrainingObjectives.reinforce`
            # TODO: The output of reinforce mismatches on some platform
            #       if variance_reduction == True
            # REINFORCE requires additional moving average variable, causing
            # it very hard to ensure two calls should have identical outputs.
            # So we disable such tests for the time being.
            with tf.variable_scope(None, default_name='reinforce'):
                # reinforce requires extra variables, but ZhuSuan does not
                # obtain a dedicated variable scope.  so we open one here.
                zs_reinforce = zs_obj.reinforce(variance_reduction=False)
            vi_reinforce = vi.training.reinforce(variance_reduction=False)
            ensure_variables_initialized()
            np.testing.assert_allclose(*sess.run([zs_reinforce, vi_reinforce]))

    def test_importance_weighted_objective(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(tf.constant(0.), [tf.constant(0.)],
                                  axis=None)
        with pytest.raises(
                ValueError, match='importance weighted lower-bound requires '
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

        # test with sampling axis
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

            # test :meth:`VariationalLowerBounds.importance_weighted_objective`
            np.testing.assert_allclose(*sess.run(
                [zs_obj, vi.lower_bound.importance_weighted_objective()]))

            # test :meth:`VariationalTrainingObjectives.iwae`
            np.testing.assert_allclose(
                *sess.run([zs_obj.sgvb(), vi.training.iwae()]))

            # test :meth:`VariationalTrainingObjectives.vimco`
            np.testing.assert_allclose(
                *sess.run([zs_obj.vimco(), vi.training.vimco()]))

    def test_klpq(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(tf.constant(0.), [tf.constant(0.)],
                                  axis=None)
        with pytest.raises(
                ValueError, match='reweighted wake-sleep training objective '
                                  'requires multi-samples'):
            _ = vi.training.rws_wake()

        # test with sampling axis
        with self.test_session() as sess:
            prepared = self.prepare_model(zs.variational.klpq, axis=0, n_z=7)
            vi = prepared.vi
            zs_obj = prepared.zs_obj

            # test :meth:`VariationalInference.zs_objective`
            vi_obj = prepared.vi.zs_objective(zs.variational.klpq)
            self.assertIsInstance(vi_obj, zs.variational.InclusiveKLObjective)

            # test :meth:`VariationalInference.zs_importance_weighted_objective`
            vi_obj = prepared.vi.zs_klpq()
            self.assertIsInstance(vi_obj, zs.variational.InclusiveKLObjective)

            # test :meth:`VariationalTrainingObjectives.rws_wake`
            np.testing.assert_allclose(
                *sess.run([zs_obj.rws(), vi.training.rws_wake()]))
