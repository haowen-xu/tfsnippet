import numpy as np
import tensorflow as tf

from tfsnippet.layers.convolutional.utils import get_deconv_output_length
from tfsnippet.utils import get_static_shape


class GetDeconv2dOutputLengthTestCase(tf.test.TestCase):

    def test_output_length(self):
        def check(input_size, kernel_size, strides, padding):
            output_size = get_deconv_output_length(
                input_size, kernel_size, strides, padding)
            self.assertGreater(output_size, 0)

            # assert input <- output
            x = tf.nn.conv2d(
                np.zeros([1, output_size, output_size, 1], dtype=np.float32),
                filter=np.zeros([kernel_size, kernel_size, 1, 1]),
                strides=[1, strides, strides, 1],
                padding=padding.upper(),
                data_format='NHWC'
            )
            h, w = get_static_shape(x)[1:3]
            self.assertEqual(input_size, h)

        check(input_size=7, kernel_size=1, strides=1, padding='same')
        check(input_size=7, kernel_size=1, strides=1, padding='valid')

        check(input_size=7, kernel_size=2, strides=1, padding='same')
        check(input_size=7, kernel_size=2, strides=1, padding='valid')
        check(input_size=7, kernel_size=1, strides=2, padding='same')
        check(input_size=7, kernel_size=1, strides=2, padding='valid')

        check(input_size=7, kernel_size=3, strides=1, padding='same')
        check(input_size=7, kernel_size=3, strides=1, padding='valid')
        check(input_size=7, kernel_size=1, strides=3, padding='same')
        check(input_size=7, kernel_size=1, strides=3, padding='valid')

        check(input_size=7, kernel_size=2, strides=3, padding='same')
        check(input_size=7, kernel_size=2, strides=3, padding='valid')
        check(input_size=7, kernel_size=3, strides=2, padding='same')
        check(input_size=7, kernel_size=3, strides=2, padding='valid')
