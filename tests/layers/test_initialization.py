import tensorflow as tf

from tfsnippet.layers import *


class DefaultKernelInitializerTestCase(tf.test.TestCase):

    def test_default_kernel_initializer(self):
        # test the default case
        i = default_kernel_initializer()
        self.assertIn('GlorotNormal', str(type(i)))
        # self.assertIsInstance(i, tf.glorot_normal_initializer) # py2 will fail

        # test the case to use weight normalization
        i = default_kernel_initializer(weight_norm=True)
        self.assertIn('RandomNormal', str(type(i)))
        # self.assertIsInstance(i, tf.random_normal_initializer) # py2 will fail
        self.assertEqual(i.stddev, .05)

        # test the case not to use weight normalization
        i = default_kernel_initializer(weight_norm=False)
        self.assertIn('GlorotNormal', str(type(i)))
        # self.assertIsInstance(i, tf.glorot_normal_initializer) # py2 will fail
