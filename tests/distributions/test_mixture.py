import numpy as np
import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet import Categorical, Normal, Mixture, OnehotCategorical
from tfsnippet.utils import set_random_seed


class MixtureTestCase(tf.test.TestCase):

    def test_errors(self):
        with pytest.raises(TypeError,
                           match='`categorical` must be a Categorical '
                                 'distribution'):
            _ = Mixture(Normal(0., 0.), [Normal(0., 0.)])

        with pytest.raises(ValueError,
                           match='Dynamic `categorical.n_categories` is not '
                                 'supported'):
            _ = Mixture(Categorical(logits=tf.placeholder(tf.float32, [None])),
                        [Normal(0., 0.)])

        with pytest.raises(ValueError, match='`components` must not be empty'):
            _ = Mixture(Categorical(logits=tf.zeros([5])), [])

        with pytest.raises(ValueError,
                           match=r'`len\(components\)` != `categorical.'
                                 r'n_categories`: 1 vs 5'):
            _ = Mixture(Categorical(logits=tf.zeros([5])), [Normal(0., 0.)])

        with pytest.raises(ValueError,
                           match='`dtype` of the 1-th component does not '
                                 'agree with the first component'):
            _ = Mixture(Categorical(logits=tf.zeros([2])),
                        [Categorical(tf.zeros([2, 3]), dtype=tf.int32),
                         Categorical(tf.zeros([2, 3]), dtype=tf.float32)])

        with pytest.raises(ValueError,
                           match='`value_ndims` of the 1-th component does not '
                                 'agree with the first component'):
            _ = Mixture(Categorical(logits=tf.zeros([2])),
                        [Categorical(tf.zeros([2, 3])),
                         OnehotCategorical(tf.zeros([2, 3]))])

        with pytest.raises(ValueError,
                           match='`is_continuous` of the 1-th component does '
                                 'not agree with the first component'):
            _ = Mixture(Categorical(logits=tf.zeros([2])),
                        [Categorical(tf.zeros([2, 3]), dtype=tf.float32),
                         Normal(tf.zeros([2]), tf.zeros([2]))])

        with pytest.raises(ValueError,
                           match='the 0-th component is not re-parameterized'):
            _ = Mixture(Categorical(logits=tf.zeros([2])),
                        [Categorical(tf.zeros([2, 3]), dtype=tf.float32),
                         Normal(tf.zeros([2]), tf.zeros([2]))],
                        is_reparameterized=True)

        with pytest.raises(RuntimeError,
                           match='.* is not re-parameterized'):
            m = Mixture(
                Categorical(logits=tf.zeros([2])),
                [Normal(-1., 0.), Normal(1., 0.)]
            )
            _ = m.sample(1, is_reparameterized=True)

        with pytest.raises(ValueError,
                           match='Batch shape of `categorical` does not '
                                 'agree with the first component'):
            _ = Mixture(
                Categorical(logits=tf.zeros([1, 3, 2])),
                [Normal(mean=tf.zeros([3]), logstd=0.),
                 Normal(mean=tf.zeros([3]), logstd=0.)]
            )

        with pytest.raises(ValueError,
                           match='Batch shape of the 1-th component does not '
                                 'agree with the first component'):
            _ = Mixture(
                Categorical(logits=tf.zeros([3, 2])),
                [Normal(mean=tf.zeros([3]), logstd=0.),
                 Normal(mean=tf.zeros([4]), logstd=0.)]
            )

    def do_check_mixture(self, component_factory, value_ndims, batch_shape,
                         is_continuous, dtype, logits_dtype,
                         is_reparameterized):
        def make_distributions(n_samples, compute_density=False):
            logits = np.random.normal(size=batch_shape + [3])
            logits = logits.astype(logits_dtype)
            categorical = Categorical(logits=logits)
            components = [
                component_factory(), component_factory(), component_factory()
            ]
            cat_sample = categorical.sample(
                n_samples, compute_density=compute_density)
            c_samples = [c.sample(n_samples, compute_density=compute_density)
                         for c in components]
            categorical.sample = Mock(return_value=cat_sample)
            for c, c_sample in zip(components, c_samples):
                c.sample = Mock(return_value=c_sample)
            return categorical, components, cat_sample, c_samples

        def check_sampling():
            t = mixture.sample(n_samples)
            out = sess.run([t, cat_sample] + list(c_samples))
            m_sample, cat = out[:2]
            component_samples = out[2:]
            samples_stack = np.stack(component_samples, axis=-value_ndims - 1)
            mask = np.eye(mixture.n_components, mixture.n_components)[cat]
            mask = np.reshape(mask, mask.shape + (1,) * value_ndims)
            ans = np.sum(mask * samples_stack, axis=-value_ndims - 1)
            np.testing.assert_allclose(m_sample, ans)

        def log_sum_exp(x, axis, keepdims=False):
            x_max = np.max(x, axis=axis, keepdims=True)
            ret = x_max + np.log(
                np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
            if not keepdims:
                ret = np.squeeze(ret, axis=axis)
            return ret

        def get_log_prob(t, group_ndims=0):
            cat_log_probs = [
                np.reshape(x, x.shape[:-1])
                for x in np.split(
                    sess.run(tf.nn.log_softmax(categorical.logits)),
                    mixture.n_components,
                    axis=-1
                )
            ]
            c_log_probs = sess.run([c.log_prob(t) for c in components])
            log_prob = log_sum_exp(
                np.stack(
                    [cat + c for cat, c in zip(cat_log_probs, c_log_probs)],
                    axis=0
                ),
                axis=0
            )
            if group_ndims > 0:
                log_prob = np.sum(
                    log_prob,
                    axis=tuple(range(-group_ndims, 0))
                )
            return log_prob

        def check_prob(group_ndims):
            t = mixture.sample(n_samples)
            t, log_prob, prob = sess.run([
                t,
                t.log_prob(group_ndims=group_ndims),
                t.prob(group_ndims=group_ndims)
            ])
            np.testing.assert_allclose(
                get_log_prob(t, group_ndims), log_prob,
                rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                np.exp(get_log_prob(t, group_ndims)), prob,
                rtol=1e-5, atol=1e-6
            )

        def check_sample_group_ndims(group_ndims, compute_density=None):
            t = mixture.sample(n_samples, group_ndims=group_ndims,
                               compute_density=compute_density)
            t, log_prob, prob = sess.run([t, t.log_prob(), t.prob()])
            np.testing.assert_allclose(
                get_log_prob(t, group_ndims), log_prob,
                rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                np.exp(get_log_prob(t, group_ndims)), prob,
                rtol=1e-5, atol=1e-6
            )

        set_random_seed(1234)

        with self.test_session() as sess:
            n_samples = 11

            categorical, components, cat_sample, c_samples = \
                make_distributions(n_samples)
            mixture = Mixture(categorical, components,
                              is_reparameterized=is_reparameterized)

            self.assertIs(mixture.categorical, categorical)
            self.assertTupleEqual(mixture.components, tuple(components))
            self.assertEqual(mixture.n_components, 3)
            self.assertEqual(mixture.dtype, dtype)
            self.assertEqual(mixture.is_continuous, is_continuous)
            self.assertEqual(mixture.is_reparameterized, is_reparameterized)
            self.assertEqual(mixture.value_ndims, value_ndims)

            check_sampling()

            check_prob(0)
            check_prob(1)

            check_sample_group_ndims(0)
            check_sample_group_ndims(1)
            check_sample_group_ndims(0, compute_density=False)
            check_sample_group_ndims(1, compute_density=False)
            check_sample_group_ndims(0, compute_density=True)
            check_sample_group_ndims(1, compute_density=True)

    def test_value_ndims_0(self):
        self.do_check_mixture(
            lambda: Normal(
                mean=np.random.normal(size=[4, 5]).astype(np.float64),
                logstd=np.random.normal(size=[4, 5]).astype(np.float64)
            ),
            value_ndims=0,
            batch_shape=[4, 5],
            is_continuous=True,
            dtype=tf.float64,
            logits_dtype=np.float64,
            is_reparameterized=True
        )

    def test_value_ndims_1(self):
        self.do_check_mixture(
            lambda: OnehotCategorical(
                logits=np.random.normal(size=[4, 5, 7]).astype(np.float32),
                dtype=tf.int32
            ),
            value_ndims=1,
            batch_shape=[4, 5],
            is_continuous=False,
            dtype=tf.int32,
            logits_dtype=np.float32,
            is_reparameterized=False
        )
