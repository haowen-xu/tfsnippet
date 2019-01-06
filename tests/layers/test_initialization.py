import tensorflow as tf

from tfsnippet.layers import *
from tfsnippet.utils import *


class DefaultKernelInitializerTestCase(tf.test.TestCase):

    def test_default_kernel_initializer(self):
        i = default_kernel_initializer(weight_norm=True)
        self.assertEqual(i.stddev, .05)

        i = default_kernel_initializer(weight_norm=(lambda t: t))
        self.assertEqual(i.stddev, .05)

        i = default_kernel_initializer(weight_norm=False)
        self.assertFalse(hasattr(i, 'stddev'))

        i = default_kernel_initializer(weight_norm=None)
        self.assertFalse(hasattr(i, 'stddev'))


class GetVariableDdiTestCase(tf.test.TestCase):

    def test_get_variable_ddi(self):
        @global_reuse
        def f(initial_value, initializing=False):
            return get_variable_ddi(
                'x', initial_value, shape=(), initializing=initializing)

        with self.test_session() as sess:
            x_in = tf.placeholder(dtype=tf.float32, shape=())
            x = f(x_in, initializing=True)
            self.assertEqual(sess.run(x, feed_dict={x_in: 123.}), 123.)
            x = f(x_in, initializing=False)
            self.assertEqual(sess.run(x, feed_dict={x_in: 456.}), 123.)
