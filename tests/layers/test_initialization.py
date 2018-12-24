import tensorflow as tf

from tfsnippet.layers import *


class DefaultKernelInitializerTestCase(tf.test.TestCase):

    def test_default_kernel_initializer(self):
        i = default_kernel_initializer(weight_norm=True)
        self.assertEqual(i.stddev, .05)
