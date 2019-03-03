import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Distribution, reduce_group_ndims


class DistributionTestCase(tf.test.TestCase):

    def test_basic(self):
        class _Distribution(Distribution):
            def log_prob(self, given, group_ndims=0, name=None):
                return reduce_group_ndims(
                    tf.reduce_sum,
                    tf.convert_to_tensor(given) - 1.,
                    group_ndims
                )

        with self.test_session() as sess:
            distrib = _Distribution(
                dtype=tf.float32,
                is_reparameterized=True,
                is_continuous=True,
                batch_shape=tf.constant([]),
                batch_static_shape=tf.TensorShape([]),
                value_ndims=0,
            )
            self.assertIs(distrib.base_distribution, distrib)
            x = np.asarray([0., 1., 2.])
            np.testing.assert_allclose(
                sess.run(distrib.prob(x, group_ndims=0)),
                np.exp(x - 1.)
            )
            np.testing.assert_allclose(
                sess.run(distrib.prob(x, group_ndims=1)),
                np.exp(np.sum(x - 1., -1))
            )
