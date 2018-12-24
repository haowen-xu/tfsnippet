import tensorflow as tf

from tfsnippet.layers import *


class DefaultKernelInitializerTestCase(tf.test.TestCase):

    def test_default_kernel_initializer(self):
        # test the default case
        i = default_kernel_initializer()
        self.assertIsInstance(i, tf.glorot_normal_initializer)

        # test the case to use weight normalization
        i = default_kernel_initializer(weight_norm=True)
        self.assertIsInstance(i, tf.random_normal_initializer)
        self.assertEqual(i.stddev, .05)

        # test the case not to use weight normalization
        i = default_kernel_initializer(weight_norm=False)
        self.assertIsInstance(i, tf.glorot_normal_initializer)
