import numpy as np
import tensorflow as tf
from mock import mock, Mock

from tfsnippet.layers import dropout


class DropoutTestCase(tf.test.TestCase):

    def test_dropout(self):
        with self.test_session() as sess:
            x = np.random.normal(size=[2, 3, 4, 5]).astype(np.float64)

            # test noise_shape = None
            noise = np.random.uniform(size=x.shape, low=0., high=1.)
            mask = (noise <= .6).astype(np.float64)
            noise_tensor = tf.convert_to_tensor(noise)

            with mock.patch('tensorflow.random_uniform',
                            Mock(return_value=noise_tensor)) as m:
                # training = True
                y = dropout(x, rate=0.4, training=True)
                self.assertDictEqual(dict(m.call_args[1]), {
                    'shape': (2, 3, 4, 5),
                    'dtype': tf.float64,
                    'minval': 0.,
                    'maxval': 1.,
                })
                np.testing.assert_allclose(sess.run(y), x * mask / .6)
                m.reset_mock()

                # training = False
                y = dropout(x, rate=0.4, training=False)
                self.assertFalse(m.called)
                np.testing.assert_allclose(sess.run(y), x)

            # test specify noise shape, and dynamic training
            noise = np.random.uniform(size=[3, 1, 5], low=0., high=1.)
            mask = (noise <= .4).astype(np.float64)
            noise_tensor = tf.convert_to_tensor(noise)
            training = tf.placeholder(dtype=tf.bool, shape=())

            with mock.patch('tensorflow.random_uniform',
                            Mock(return_value=noise_tensor)) as m:
                y = dropout(x, rate=tf.constant(0.6, dtype=tf.float32),
                            training=training, noise_shape=(3, 1, 5))
                self.assertDictEqual(dict(m.call_args[1]), {
                    'shape': (3, 1, 5),
                    'dtype': tf.float64,
                    'minval': 0.,
                    'maxval': 1.,
                })

            np.testing.assert_allclose(
                sess.run(y, feed_dict={training: True}),
                x * mask / .4
            )
            np.testing.assert_allclose(
                sess.run(y, feed_dict={training: False}), x)
