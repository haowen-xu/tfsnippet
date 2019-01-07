import tensorflow as tf

from tfsnippet.layers import *


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
