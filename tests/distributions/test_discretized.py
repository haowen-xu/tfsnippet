import functools

import numpy as np
import pytest
import tensorflow as tf
from mock import mock, Mock

from tfsnippet import DiscretizedLogistic


def safe_sigmoid(x):
    return np.where(x < 0, np.exp(x) / (1. + np.exp(x)), 1. / (1. + np.exp(-x)))


def discretize(x, bin_size, min_val=None, max_val=None):
    if min_val is not None:
        x = x - min_val
    x = np.floor(x / bin_size + .5) * bin_size
    if min_val is not None:
        x = x + min_val

    if min_val is not None:
        x = np.maximum(x, min_val)
    if max_val is not None:
        x = np.minimum(x, max_val)

    return x


def naive_discretized_logistic_pdf(
        x, mean, log_scale, bin_size, min_val=None, max_val=None,
        biased_edges=True, discretize_given=True, group_ndims=0):
    # discretize x
    if discretize_given:
        x = discretize(x, bin_size, min_val, max_val)

    # middle pdfs
    x_hi = (x - mean + bin_size * 0.5) / np.exp(log_scale)
    x_low = (x - mean - bin_size * 0.5) / np.exp(log_scale)
    cdf_delta = safe_sigmoid(x_hi) - safe_sigmoid(x_low)
    middle_pdf = np.log(np.maximum(cdf_delta, 1e-7))
    log_prob = middle_pdf

    # left edge
    if min_val is not None and biased_edges:
        log_prob = np.where(
            x < min_val + bin_size * 0.5,
            np.log(safe_sigmoid(x_hi)),
            log_prob
        )

    # right edge
    if max_val is not None and biased_edges:
        log_prob = np.where(
            x >= max_val - bin_size * 0.5,
            np.log(1. - safe_sigmoid(x_low)),
            log_prob
        )

    if group_ndims > 0:
        log_prob = np.sum(log_prob, tuple(range(-group_ndims, 0)))
    return log_prob


def naive_discretized_logistic_sample(
        uniform_samples, mean, log_scale, bin_size, min_val=None, max_val=None,
        discretize_sample=True):
    u = uniform_samples
    samples = mean + np.exp(log_scale) * (np.log(u) - np.log(1. - u))
    if discretize_sample:
        samples = discretize(samples, bin_size, min_val, max_val)
    return samples


class DiscretizedLogisticTestCase(tf.test.TestCase):

    def test_props(self):
        with self.test_session() as sess:
            mean = np.random.normal(size=[2, 1, 4]).astype(np.float32)
            log_scale = np.random.normal(size=[3, 1, 5, 1]).astype(np.float32)

            d = DiscretizedLogistic(
                mean=mean,
                log_scale=log_scale,
                bin_size=1. / 255,
                min_val=-1.,
                max_val=2.,
                biased_edges=False
            )
            self.assertEqual(d.dtype, tf.float32)
            self.assertFalse(d.is_continuous)
            self.assertFalse(d.is_reparameterized)
            self.assertEqual(d.value_ndims, 0)
            self.assertEqual(d.get_batch_shape().as_list(), [3, 2, 5, 4])
            self.assertEqual(list(sess.run(d.batch_shape)), [3, 2, 5, 4])
            self.assertEqual(d.bin_size.dtype, tf.float32)
            self.assertFalse(d.biased_edges)

            np.testing.assert_allclose(sess.run(d.mean), mean)
            np.testing.assert_allclose(sess.run(d.log_scale), log_scale)
            np.testing.assert_allclose(sess.run(d.bin_size), 1. / 255)
            np.testing.assert_allclose(sess.run(d.min_val), -1.)
            np.testing.assert_allclose(sess.run(d.max_val), 2.)

            d = DiscretizedLogistic(mean=mean, log_scale=log_scale, bin_size=1.,
                                    dtype=tf.int32)
            self.assertEqual(d.dtype, tf.int32)
            self.assertIsNone(d.min_val)
            self.assertIsNone(d.max_val)
            self.assertTrue(d.biased_edges)

    def test_errors(self):
        with pytest.raises(ValueError,
                           match='`bin_size` is a float number, but `dtype` '
                                 'is not a float number type'):
            _ = DiscretizedLogistic(tf.zeros([]), tf.zeros([]), 0.1,
                                    dtype=tf.int32)

        with pytest.raises(ValueError,
                           match='`min_val` and `max_val` must be both None '
                                 'or neither None.'):
            _ = DiscretizedLogistic(tf.zeros([2, 3]), 0., bin_size=.1,
                                    min_val=-1.)

        with pytest.raises(ValueError,
                           match='`min_val` and `max_val` must be both None '
                                 'or neither None.'):
            _ = DiscretizedLogistic(tf.zeros([2, 3]), 0., bin_size=.1,
                                    max_val=1.)

        with pytest.raises(ValueError,
                           match='`min_val - max_val` must be multiples of '
                                 '`bin_size`: max_val - min_val = 1.5 vs '
                                 'bin_size = 1.'):
            _ = DiscretizedLogistic(tf.zeros([2, 3]), 0., bin_size=1.0,
                                    min_val=-0.5, max_val=1.)

        with pytest.raises(ValueError,
                           match='The shape of `mean` and `log_scale` cannot '
                                 'be broadcasted'):
            _ = DiscretizedLogistic(tf.zeros([2]), tf.zeros([3]), 0.1)

    def test_sample(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        np.random.seed(1234)
        mean = 3 * np.random.uniform(size=[2, 1, 4]).astype(np.float64) - 1
        log_scale = np.random.normal(size=[3, 1, 5, 1]).astype(np.float64)
        bin_size = 1 / 255.
        epsilon = 1e-7

        with self.test_session() as sess:
            # n_samples == None
            min_val = -1.5
            max_val = 1.5
            n_samples = None
            sample_shape = [3, 2, 5, 4]
            u = np.random.uniform(low=epsilon, high=1. - epsilon,
                                  size=sample_shape).astype(np.float64)
            x_ans = naive_discretized_logistic_sample(
                u, mean, log_scale, bin_size, min_val, max_val)

            def patched_rnd_uniform(*args, **kwargs):
                return tf.convert_to_tensor(u)

            with mock.patch('tensorflow.random_uniform',
                            Mock(wraps=patched_rnd_uniform)) as rnd_uniform:
                d = DiscretizedLogistic(
                    mean=mean, log_scale=log_scale, bin_size=bin_size,
                    min_val=min_val, max_val=max_val, biased_edges=True,
                    dtype=tf.float64
                )
                x = d.sample(n_samples=n_samples, compute_density=True)
                args = rnd_uniform.call_args[1]
                self.assertEqual(list(sess.run(args['shape'])), sample_shape)
                self.assertEqual(args['minval'], epsilon)
                self.assertEqual(args['maxval'], 1. - epsilon)
                self.assertEqual(args['dtype'], tf.float64)
                assert_allclose(sess.run(x), x_ans)
            assert_allclose(
                sess.run(x.log_prob()),
                naive_discretized_logistic_pdf(
                    x_ans, mean, log_scale, bin_size, min_val, max_val,
                    group_ndims=0
                )
            )

            # n_samples == 11, min_val = max_val = None
            min_val = None
            max_val = None
            n_samples = 11
            sample_shape = [11, 3, 2, 5, 4]
            u = np.random.uniform(low=epsilon, high=1. - epsilon,
                                  size=sample_shape).astype(np.float64)
            x_ans = naive_discretized_logistic_sample(
                u, mean, log_scale, bin_size, min_val, max_val,
                discretize_sample=False
            )

            def patched_rnd_uniform(*args, **kwargs):
                return tf.convert_to_tensor(u)

            with mock.patch('tensorflow.random_uniform',
                            Mock(wraps=patched_rnd_uniform)) as rnd_uniform:
                d = DiscretizedLogistic(
                    mean=mean, log_scale=log_scale, bin_size=bin_size,
                    min_val=min_val, max_val=max_val, biased_edges=True,
                    dtype=tf.float64, discretize_sample=False,
                    discretize_given=False
                )
                x = d.sample(n_samples=n_samples, group_ndims=1)
                args = rnd_uniform.call_args[1]
                self.assertEqual(list(sess.run(args['shape'])), sample_shape)
                self.assertEqual(args['minval'], epsilon)
                self.assertEqual(args['maxval'], 1. - epsilon)
                self.assertEqual(args['dtype'], tf.float64)
                assert_allclose(sess.run(x), x_ans)
            assert_allclose(
                sess.run(x.log_prob()),
                naive_discretized_logistic_pdf(
                    x_ans, mean, log_scale, bin_size, min_val, max_val,
                    group_ndims=1, discretize_given=False
                )
            )

    def test_log_prob(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        np.random.seed(1234)
        x = np.random.normal(size=[7, 3, 2, 5, 4]).astype(np.float64)
        self.assertLess(np.min(x), -1.1)
        self.assertGreater(np.max(x), 2.1)
        self.assertGreater(np.sum(np.logical_and(x > -0.5, x < 1.5)), 0)

        mean = 3 * np.random.uniform(size=[2, 1, 4]).astype(np.float64) - 1
        log_scale = np.random.normal(size=[3, 1, 5, 1]).astype(np.float64)
        bin_size = 1 / 256.
        min_val = -1.
        max_val = 2.

        with self.test_session() as sess:
            # biased_edges = False, discretize_given = True
            d = DiscretizedLogistic(
                mean=mean, log_scale=log_scale, bin_size=bin_size,
                min_val=None, max_val=None, biased_edges=False,
                dtype=tf.float64
            )

            assert_allclose(
                sess.run(d.log_prob(x, group_ndims=0)),
                naive_discretized_logistic_pdf(
                    x, mean, log_scale, bin_size, None, None,
                    biased_edges=False, group_ndims=0)
            )

            # biased_edges = False, discretize_given = True
            d = DiscretizedLogistic(
                mean=mean, log_scale=log_scale, bin_size=bin_size,
                min_val=min_val, max_val=max_val, biased_edges=False,
                dtype=tf.float64
            )

            assert_allclose(
                sess.run(d.log_prob(x, group_ndims=0)),
                naive_discretized_logistic_pdf(
                    x, mean, log_scale, bin_size, min_val, max_val,
                    biased_edges=False, group_ndims=0)
            )

            # biased_edges = False, discretize_given = False
            d = DiscretizedLogistic(
                mean=mean, log_scale=log_scale, bin_size=bin_size,
                min_val=min_val, max_val=max_val, biased_edges=False,
                dtype=tf.float64, discretize_given=False
            )

            assert_allclose(
                sess.run(d.log_prob(x, group_ndims=0)),
                naive_discretized_logistic_pdf(
                    x, mean, log_scale, bin_size, min_val, max_val,
                    biased_edges=False, discretize_given=False, group_ndims=0)
            )

            # biased_edges = True, discretize_given = True
            d = DiscretizedLogistic(
                mean=mean, log_scale=log_scale, bin_size=bin_size,
                min_val=min_val, max_val=max_val, biased_edges=True,
                dtype=tf.float64
            )

            assert_allclose(
                sess.run(d.log_prob(x, group_ndims=0)),
                naive_discretized_logistic_pdf(
                    x, mean, log_scale, bin_size, min_val, max_val,
                    biased_edges=True, group_ndims=0)
            )

            assert_allclose(
                sess.run(d.log_prob(x, group_ndims=2)),
                naive_discretized_logistic_pdf(
                    x, mean, log_scale, bin_size, min_val, max_val,
                    biased_edges=True, group_ndims=2)
            )

    def test_prob_sum(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        np.random.seed(1234)
        x = np.reshape(np.linspace(0, 1, 11), [-1, 1, 1, 1, 1])
        mean = np.random.uniform(size=[2, 1, 4]).astype(np.float64)
        log_scale = np.random.normal(size=[3, 1, 5, 1]).astype(np.float64)
        bin_size = 1 / 10.
        min_val = 0.
        max_val = 1.

        with self.test_session() as sess:

            d = DiscretizedLogistic(
                mean=mean, log_scale=log_scale, bin_size=bin_size,
                min_val=min_val, max_val=max_val, biased_edges=True,
                dtype=tf.float64
            )
            prob = sess.run(tf.reduce_sum(d.prob(x), 0))
            self.assertEqual(prob.shape, (3, 2, 5, 4))

            assert_allclose(prob, np.ones_like(prob))

    def test_log_prob_extreme(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        x = np.asarray(0., dtype=np.float64)
        mean = np.asarray(0., dtype=np.float64)
        bin_size = 1 / 256.

        # to ensure bin_size / (2*scale) < cdf_delta
        cdf_delta = 1e-8
        s = np.linspace(0, 20, 101)
        t = bin_size / (2 * np.exp(s))
        idx = np.where(safe_sigmoid(t) - safe_sigmoid(-t) < cdf_delta)[0]
        self.assertGreater(np.size(idx), 2)
        log_scale = s[idx]

        with self.test_session() as sess:
            # now compute the log-probability of this extreme case
            d = DiscretizedLogistic(
                mean=mean, log_scale=log_scale, bin_size=bin_size,
                epsilon=1e-7
            )
            assert_allclose(
                sess.run(d.log_prob(x, group_ndims=0)),
                naive_discretized_logistic_pdf(
                    x, mean, log_scale, bin_size, None, None,
                    biased_edges=True, group_ndims=0)
            )

            # now compute the log-probability of this extreme case,
            # but discretize_given = False
            d = DiscretizedLogistic(
                mean=mean, log_scale=log_scale, bin_size=bin_size,
                epsilon=1e-7, discretize_given=False
            )
            assert_allclose(
                sess.run(d.log_prob(x, group_ndims=0)),
                naive_discretized_logistic_pdf(
                    x, mean, log_scale, bin_size, None, None,
                    biased_edges=True, discretize_given=False, group_ndims=0)
            )
