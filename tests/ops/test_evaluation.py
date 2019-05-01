import numpy as np
import tensorflow as tf

from tfsnippet.ops import bits_per_dimension


class BitsPerDimensionTestCase(tf.test.TestCase):

    def test_bits_per_dimension(self):
        with self.test_session() as sess:
            log_p = np.random.normal(size=[2, 3, 4, 5])

            np.testing.assert_allclose(
                sess.run(bits_per_dimension(log_p, 1., scale=None)),
                -log_p / np.log(2)
            )
            np.testing.assert_allclose(
                sess.run(bits_per_dimension(log_p, 1024 * 3, scale=None)),
                -log_p / (np.log(2) * 1024 * 3)
            )
            np.testing.assert_allclose(
                sess.run(bits_per_dimension(log_p, 1., scale=256.)),
                -(log_p - np.log(256)) / np.log(2)
            )
            np.testing.assert_allclose(
                sess.run(bits_per_dimension(log_p, 1024 * 3, scale=256)),
                -(log_p - np.log(256) * 1024 * 3) / (np.log(2) * 1024 * 3)
            )
