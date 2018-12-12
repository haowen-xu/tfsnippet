import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Normal, Distribution, reduce_group_ndims


class DistributionTestCase(tf.test.TestCase):

    def test_factory(self):
        with self.test_session() as sess:
            factory = Normal.factory(std=2.)

            normal = factory(mean=1.)
            np.testing.assert_equal(
                sess.run([normal.mean, normal.std]), [1., 2.])

            # override `std`
            normal = factory(mean=2., std=3.)
            np.testing.assert_equal(
                sess.run([normal.mean, normal.std]), [2., 3.])

            # parameters can also be specified via a dict
            normal = factory({'mean': 2., 'std': 3.})
            np.testing.assert_equal(
                sess.run([normal.mean, normal.std]), [2., 3.])

            # named arguments override dict arguments
            normal = factory({'mean': 2., 'std': 3.}, std=4.)
            np.testing.assert_equal(
                sess.run([normal.mean, normal.std]), [2., 4.])

    def test_prob(self):
        class _Distribution(Distribution):
            def log_prob(self, given, group_ndims=0, name=None):
                return reduce_group_ndims(
                    tf.reduce_sum,
                    tf.convert_to_tensor(given) - 1.,
                    group_ndims
                )

        with self.test_session() as sess:
            distrib = _Distribution()
            x = np.asarray([0., 1., 2.])
            np.testing.assert_allclose(
                sess.run(distrib.prob(x, group_ndims=0)),
                np.exp(x - 1.)
            )
            np.testing.assert_allclose(
                sess.run(distrib.prob(x, group_ndims=1)),
                np.exp(np.sum(x - 1., -1))
            )
