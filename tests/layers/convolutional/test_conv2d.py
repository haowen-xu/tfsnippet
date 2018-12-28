import numpy as np
import pytest
import tensorflow as tf
from mock import mock

from tests.layers.helper import l2_normalize
from tfsnippet.layers import conv2d
from tfsnippet.utils import flatten, unflatten, is_integer

tf_conv2d = tf.nn.conv2d


def patched_conv2d(input, filter, strides, padding, data_format, dilations):
    """A patched version of `tf.nn.conv2d`, which emulates NCHW by NHWC."""
    if data_format == 'NCHW':
        transpose_axis = (0, 2, 3, 1)
        input = tf.transpose(input, transpose_axis)
        strides = tuple(strides[i] for i in transpose_axis)
        dilations = tuple(dilations[i] for i in transpose_axis)
    output = tf_conv2d(
        input=input, filter=filter, strides=strides, padding=padding,
        data_format='NHWC', dilations=dilations
    )
    if data_format == 'NCHW':
        transpose_axis = (0, 3, 1, 2)
        output = tf.transpose(output, transpose_axis)
    return output


class Conv2dTestCase(tf.test.TestCase):

    @staticmethod
    def conv2d_ans(input, padding, kernel, bias, strides, dilations,
                   activation_fn=None, normalizer_fn=None):
        """Produce the expected answer of conv2d."""
        strides = (strides,) * 2 if is_integer(strides) else tuple(strides)
        strides = (1,) + strides + (1,)

        session = tf.get_default_session()
        input, s1, s2 = flatten(input, 4)
        padding = padding.upper()

        if dilations > 1:
            assert(not any(i > 1 for i in strides))
            output = tf.nn.atrous_conv2d(
                value=input,
                filters=kernel,
                rate=dilations,
                padding=padding
            )
        else:
            output = tf.nn.conv2d(
                input=input,
                filter=kernel,
                strides=strides,
                padding=padding,
                data_format='NHWC',
                dilations=[1] * 4
            )
        if bias is not None:
            output += bias
        if normalizer_fn:
            output = normalizer_fn(output)
        if activation_fn:
            output = activation_fn(output)

        output = unflatten(output, s1, s2)
        output = session.run(output)
        return output

    @staticmethod
    def run_conv2d(input, filters, kernel_size, padding, kernel, bias,
                   strides, dilations, channels_last, ph=None, **kwargs):
        """Run `tfsnippet.layers.conv2d` and get the output."""
        i_shape = input.shape
        if not channels_last:
            input = np.transpose(
                input,
                tuple(i for i in range(len(i_shape) - 3)) + (-1, -3, -2)
            )
            if bias is not None:
                assert(len(bias.shape) == 1)
                bias = np.reshape(bias, (bias.shape[0], 1, 1))

        session = tf.get_default_session()
        output = session.run(
            conv2d(
                input=ph if ph is not None else input,
                filters=filters,
                kernel_size=kernel_size,
                channels_last=channels_last,
                padding=padding,
                strides=strides,
                dilations=dilations,
                kernel=kernel,
                bias=bias,
                **kwargs
            ),
            feed_dict={ph: input} if ph is not None else None
        )
        if not channels_last:
            output = np.transpose(
                output,
                tuple(i for i in range(len(i_shape) - 3)) + (-2, -1, -3)
            )
        return output

    def test_conv2d(self):
        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 32, 5]).astype(np.float32)
            kernel = np.random.random(size=[3, 4, 5, 7]).astype(np.float32)
            bias = np.random.random(size=[7]).astype(np.float32)

            # test strides 1, skip 1, same padding, NHWC
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'same', kernel, bias, 1, 1,
                                channels_last=True),
                self.conv2d_ans(x, 'same', kernel, bias, 1, 1)
            )

            # test strides 1, skip 1, valid padding, NCHW
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, (1, 1), 1,
                                channels_last=False),
                self.conv2d_ans(x, 'valid', kernel, bias, 1, 1)
            )

            # test strides (3, 2), skip 1, same padding, NHWC
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'same', kernel, bias, (3, 2), 1,
                                channels_last=True),
                self.conv2d_ans(x, 'same', kernel, bias, (3, 2), 1)
            )

            # test strides 1, skip 2, valid padding, NHWC
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, 1, 2,
                                channels_last=True),
                self.conv2d_ans(x, 'valid', kernel, bias, 1, 2)
            )

            # test dynamic shape, same padding, NHWC
            ph = tf.placeholder(dtype=tf.float32,
                                shape=(None, None, None, None, 5))
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'same', kernel, bias, 1, 1,
                                channels_last=True, ph=ph),
                self.conv2d_ans(x, 'same', kernel, bias, 1, 1)
            )

            # test dynamic shape, valid padding NCHW
            ph = tf.placeholder(dtype=tf.float32,
                                shape=(None, None, 5, None, None))
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, 1, 1,
                                channels_last=False, ph=ph),
                self.conv2d_ans(x, 'valid', kernel, bias, 1, 1)
            )

            # test errors
            with pytest.raises(ValueError,
                               match='Invalid value for argument `strides`: '
                                     'expected to be one or two positive '
                                     'integers'):
                _ = self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, 0, 2,
                                    channels_last=False)

            with pytest.raises(ValueError,
                               match='`channels_last` == False is incompatible '
                                     'with `dilations` > 1'):
                _ = self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, 1, 2,
                                    channels_last=False)

            with pytest.raises(ValueError,
                               match='`strides` > 1 is incompatible with '
                                     '`dilations` > 1'):
                _ = self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, 2, 2,
                                    channels_last=True)

        # test create variables
        with tf.Graph().as_default():
            # test NHWC
            _ = conv2d(x, 7, (3, 4), padding='same', channels_last=True)
            kernel_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-2]
            bias_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-1]
            self.assertEqual(kernel_var.shape, kernel.shape)
            self.assertTrue(kernel_var.name.endswith('/kernel:0'))
            self.assertEqual(bias_var.shape, bias.shape)
            self.assertTrue(bias_var.name.endswith('/bias:0'))

            # test NCHW
            _ = conv2d(np.transpose(x, [0, 1, -1, -3, -2]), 7, (3, 4),
                       padding='valid', channels_last=False)
            kernel_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-2]
            bias_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-1]
            self.assertEqual(kernel_var.shape, kernel.shape)
            self.assertTrue(kernel_var.name.endswith('/kernel:0'))
            self.assertEqual(bias_var.shape, bias.shape + (1, 1))
            self.assertTrue(bias_var.name.endswith('/bias:0'))

        # test create variables with use_bias = False
        with tf.Graph().as_default():
            _ = conv2d(x, 7, (3, 4), padding='same', channels_last=True,
                       use_bias=False)
            kernel_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-1]
            self.assertEqual(kernel_var.shape, kernel.shape)
            self.assertTrue(kernel_var.name.endswith('/kernel:0'))

    def test_normalization_and_activation(self):
        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 32, 5]).astype(np.float32)
            kernel = np.random.random(size=[3, 4, 5, 7]).astype(np.float32)
            normalized_kernel = l2_normalize(kernel, axis=(0, 1, 2))
            kernel = kernel.astype(np.float32)
            bias = np.random.random(size=[7]).astype(np.float32)

            normalizer_fn = lambda x: x * 1.5 - 3.
            activation_fn = lambda x: x * 2. + 1.

            # test weight_norm + normalizer + activation, NHWC
            self.assertLess(
                np.max(np.abs(
                    self.run_conv2d(x, 7, (3, 4), 'same', kernel, bias, 1, 1,
                                    channels_last=True, weight_norm=True,
                                    normalizer_fn=normalizer_fn,
                                    activation_fn=activation_fn) -
                    self.conv2d_ans(x, 'same', normalized_kernel, None, 1, 1,
                                    normalizer_fn=normalizer_fn,
                                    activation_fn=activation_fn)
                )),
                1e-5
            )

            # test weight_norm + normalizer + activation, NCHW, use_bias = True
            self.assertLess(
                np.max(np.abs(
                    self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, 1, 1,
                                    channels_last=False, weight_norm=True,
                                    use_bias=True,
                                    normalizer_fn=normalizer_fn,
                                    activation_fn=activation_fn) -
                    self.conv2d_ans(x, 'valid', normalized_kernel, bias, 1, 1,
                                    normalizer_fn=normalizer_fn,
                                    activation_fn=activation_fn)
                )),
                1e-5
            )
