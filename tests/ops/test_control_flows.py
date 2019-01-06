import tensorflow as tf

from tfsnippet.ops import *

__all__ = ['SmartCondTestCase']


class SmartCondTestCase(tf.test.TestCase):

    def test_smart_cond(self):
        with self.test_session() as sess:
            # test static condition
            self.assertEqual(1, smart_cond(True, (lambda: 1), (lambda: 2)))
            self.assertEqual(2, smart_cond(False, (lambda: 1), (lambda: 2)))

            # test dynamic condition
            cond_in = tf.placeholder(dtype=tf.bool, shape=())
            value = smart_cond(
                cond_in, lambda: tf.constant(1), lambda: tf.constant(2))
            self.assertIsInstance(value, tf.Tensor)
            self.assertEqual(sess.run(value, feed_dict={cond_in: True}), 1)
            self.assertEqual(sess.run(value, feed_dict={cond_in: False}), 2)
