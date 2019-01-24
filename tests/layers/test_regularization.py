import numpy as np
import tensorflow as tf

from tfsnippet.layers import l2_regularizer


class L2RegualrizerTestCase(tf.test.TestCase):

    def test_l2_regularizer(self):
        with self.test_session() as sess:
            w = np.random.random(size=[10, 11, 12])
            lambda_ = .75
            loss = lambda_ * .5 * np.sum(w ** 2)
            np.testing.assert_allclose(
                sess.run(l2_regularizer(lambda_)(w)),
                loss
            )

            self.assertIsNone(l2_regularizer(None))
