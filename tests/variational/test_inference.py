import functools

import numpy as np
import pytest
import tensorflow as tf

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

    def test_vimco(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(tf.constant(0.), [tf.constant(0.)],
                                  axis=None)
        with pytest.raises(
                ValueError, match='iwae training objective '
                                  'requires multi-samples'):
            _ = vi.training.iwae()

        with self.test_session() as sess:
            log_p = tf.random_normal(shape=[5, 7])
            log_q = tf.random_normal(shape=[4, 5, 7])

            vi = VariationalInference(log_p, [log_q], axis=0)
            output = vi.training.vimco()
            answer = -vimco_estimator(
                log_p - log_q, log_q, axis=0)
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
