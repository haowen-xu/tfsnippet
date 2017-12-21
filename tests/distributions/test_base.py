import numpy as np
import tensorflow as tf

from tfsnippet.distributions import Normal


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
