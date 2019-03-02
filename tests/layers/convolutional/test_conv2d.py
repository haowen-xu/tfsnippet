import functools

import numpy as np
import pytest
import tensorflow as tf
from mock import mock

from tests.helper import assert_variables
from tests.layers.convolutional.helper import *
from tests.layers.core.test_gated import safe_sigmoid
from tests.layers.helper import l2_normalize
from tfsnippet.layers import *
from tfsnippet.layers.convolutional.utils import get_deconv_output_length
from tfsnippet.ops import flatten_to_ndims, unflatten_from_ndims
from tfsnippet.utils import is_integer

tf_conv2d = tf.nn.conv2d
tf_atrous_conv2d = tf.nn.atrous_conv2d
tf_conv2d_transpose = tf.nn.conv2d_transpose
tf_atrous_conv2d_transpose = tf.nn.atrous_conv2d_transpose


def patched_conv2d(input, filter, strides, padding, data_format,
                   dilations):
    """A patched version of `tf.nn.conv2d`, emulates NCHW by NHWC."""
    input = input_maybe_to_channels_last(input, data_format=data_format)
    [strides, dilations] = strides_tuple_to_channels_last(
        [strides, dilations], data_format=data_format)
    output = tf_conv2d(
        input=input, filter=filter, strides=strides, padding=padding,
        data_format='NHWC', dilations=dilations
    )
    output = output_maybe_to_channels_first(output, data_format=data_format)
    return output


class Conv2dTestCase(tf.test.TestCase):

    @staticmethod
    def conv2d_ans(input, padding, kernel, bias, strides, dilations,
                   activation_fn=None, normalizer_fn=None, gated=False,
                   gate_sigmoid_bias=2.):
        """Produce the expected answer of conv2d."""
        strides = (strides,) * 2 if is_integer(strides) else tuple(strides)
        strides = (1,) + strides + (1,)

        session = tf.get_default_session()
        input, s1, s2 = flatten_to_ndims(input, 4)
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
        if gated:
            output, gate = tf.split(output, 2, axis=-1)
        if activation_fn:
            output = activation_fn(output)
        if gated:
            output = output * tf.sigmoid(gate + gate_sigmoid_bias)

        output = unflatten_from_ndims(output, s1, s2)
        output = session.run(output)
        return output

    @staticmethod
    def run_conv2d(input, out_channels, kernel_size, padding, kernel, bias,
                   strides, dilations, channels_last, ph=None, **kwargs):
        """Run `tfsnippet.layers.conv2d` and get the output."""
        i_shape = input.shape
        if not channels_last:
            input = np.transpose(
                input,
                tuple(i for i in range(len(i_shape) - 3)) + (-1, -3, -2)
            )

        session = tf.get_default_session()
        output = session.run(
            conv2d(
                input=ph if ph is not None else input,
                out_channels=out_channels,
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

    def test_conv2d_1x1(self):
        with self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 31, 5]).astype(np.float32)
            kernel = np.random.random(size=[1, 1, 5, 7]).astype(np.float32)
            bias = np.random.random(size=[7]).astype(np.float32)

            # test strides 1, kernel size 1, valid padding, NHWC
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, 1, 'valid', kernel, bias, 1, 1,
                                channels_last=True),
                self.conv2d_ans(x, 'valid', kernel, bias, 1, 1)
            )

            # test strides (2, 3), kernel size 1, valid padding, NHWC
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, 1, 'same', kernel, bias, (2, 3), 1,
                                channels_last=True),
                self.conv2d_ans(x, 'same', kernel, bias, (2, 3), 1)
            )

    def test_conv2d(self):
        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 31, 5]).astype(np.float32)
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
            assert_variables(['kernel', 'bias'], trainable=True, scope='conv2d',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

            kernel_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-2]
            bias_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-1]
            self.assertEqual(kernel_var.shape, kernel.shape)
            self.assertEqual(bias_var.shape, bias.shape)

            # test NCHW
            _ = conv2d(np.transpose(x, [0, 1, -1, -3, -2]), 7, (3, 4),
                       padding='valid', channels_last=False)
            assert_variables(['kernel', 'bias'], trainable=True,
                             scope='conv2d_1',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

            kernel_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-2]
            bias_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-1]
            self.assertEqual(kernel_var.shape, kernel.shape)
            self.assertEqual(bias_var.shape, bias.shape)

        # test create variables, non-trainable
        with tf.Graph().as_default():
            # test NHWC
            _ = conv2d(x, 7, (3, 4), padding='same', channels_last=True,
                       trainable=False)
            assert_variables(['kernel', 'bias'], trainable=False,
                             scope='conv2d',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test create variables with use_bias = False
        with tf.Graph().as_default():
            _ = conv2d(x, 7, (3, 4), padding='same', channels_last=True,
                       use_bias=False)
            assert_variables(['kernel'], trainable=True, scope='conv2d',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])
            assert_variables(['bias'], exist=False, scope='conv2d')

    def test_normalization_and_activation(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-5)
        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 31, 5]).astype(np.float32)
            kernel = np.random.random(size=[3, 4, 5, 7]).astype(np.float32)
            normalized_kernel = l2_normalize(kernel, axis=(0, 1, 2))
            kernel = kernel.astype(np.float32)
            bias = np.random.random(size=[7]).astype(np.float32)

            normalizer_fn = lambda x: x * 1.5 - 3.
            activation_fn = lambda x: x * 2. + 1.

            # test weight_norm + normalizer + activation, NHWC
            assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'same', kernel, bias, 1, 1,
                                channels_last=True, weight_norm=True,
                                normalizer_fn=normalizer_fn,
                                activation_fn=activation_fn),
                self.conv2d_ans(x, 'same', normalized_kernel, None, 1, 1,
                                normalizer_fn=normalizer_fn,
                                activation_fn=activation_fn)
            )

            # test weight_norm + normalizer + activation, NCHW, use_bias = True
            assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'valid', kernel, bias, 1, 1,
                                channels_last=False, weight_norm=True,
                                use_bias=True,
                                normalizer_fn=normalizer_fn,
                                activation_fn=activation_fn),
                self.conv2d_ans(x, 'valid', normalized_kernel, bias, 1, 1,
                                normalizer_fn=normalizer_fn,
                                activation_fn=activation_fn)
            )

    def test_kernel_mask(self):
        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 31, 5]).astype(np.float32)
            kernel = np.random.random(size=[3, 4, 5, 7]).astype(np.float32)
            mask = np.random.binomial(n=1, p=.5, size=kernel.shape). \
                astype(np.float32)
            bias = np.random.random(size=[7]).astype(np.float32)

            # test strides 1, skip 1, same padding, NHWC
            np.testing.assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'same', kernel, bias, 1, 1,
                                channels_last=True, kernel_mask=mask),
                self.conv2d_ans(x, 'same', kernel * mask, bias, 1, 1)
            )

    def test_gated(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-5)
        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 31, 5]).astype(np.float32)
            kernel = np.random.random(size=[3, 4, 5, 14]).astype(np.float32)
            normalized_kernel = l2_normalize(kernel, axis=(0, 1, 2))
            kernel = kernel.astype(np.float32)
            bias = np.random.random(size=[14]).astype(np.float32)

            normalizer_fn = lambda x: x * 1.5 - 3.
            activation_fn = lambda x: x * 2. + 1.

            assert_allclose(
                self.run_conv2d(x, 7, (3, 4), 'same', kernel, bias, 1, 1,
                                channels_last=True, weight_norm=True,
                                normalizer_fn=normalizer_fn,
                                activation_fn=activation_fn,
                                gated=True,
                                gate_sigmoid_bias=1.1),
                self.conv2d_ans(x, 'same', normalized_kernel, None, 1, 1,
                                normalizer_fn=normalizer_fn,
                                activation_fn=activation_fn,
                                gated=True, gate_sigmoid_bias=1.1)
            )


def patched_conv2d_transpose(value, filter, output_shape, strides, padding,
                             data_format):
    """A patched version of `tf.nn.conv2d_transpose`, emulates NCHW by NHWC."""
    value = input_maybe_to_channels_last(value, data_format=data_format)
    [strides, output_shape] = strides_tuple_to_channels_last(
        [strides, output_shape], data_format=data_format)
    output = tf_conv2d_transpose(
        value=value, filter=filter, output_shape=output_shape, strides=strides,
        padding=padding, data_format='NHWC'
    )
    output = output_maybe_to_channels_first(output, data_format=data_format)
    return output


class Deconv2dTestCase(tf.test.TestCase):

    def check(self, x, padding, kernel, bias, strides):
        """Integrated tests for specific argument combinations."""
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-5)
        strides = (strides,) * 2 if is_integer(strides) else tuple(strides)

        x_shape = (x.shape[-3], x.shape[-2])
        x_channels = x.shape[-1]
        kernel_size = kernel.shape[0], kernel.shape[1]

        # compute the input for the deconv
        y = Conv2dTestCase.conv2d_ans(x, padding, kernel, None, strides, 1)
        y_shape = (y.shape[-3], y.shape[-2])
        y_channels = y.shape[-1]

        # test explicit output_shape, NHWC
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, x_shape, padding,
            kernel, None, strides, channels_last=True, use_bias=False
        )
        self.assertEqual(deconv_out.shape, x.shape)

        # memorize the linear output for later tests
        linear_out = np.copy(deconv_out)

        # test explicit output_shape, NCHW
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, x_shape, padding,
            kernel, None, strides, channels_last=False, use_bias=False
        )
        assert_allclose(deconv_out, linear_out)

        # test explicit dynamic output_shape, NHWC
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, tf.constant(x_shape), padding,
            kernel, None, strides, channels_last=True, use_bias=False
        )
        assert_allclose(deconv_out, linear_out)

        # test explicit dynamic output_shape, NCHW
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, tf.constant(x_shape), padding,
            kernel, None, strides, channels_last=False, use_bias=False
        )
        assert_allclose(deconv_out, linear_out)

        # test dynamic input, explicit dynamic output_shape, NHWC
        ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None,) * (len(y.shape) - 3) + (None, None, y_channels)
        )
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, tf.constant(x_shape), padding,
            kernel, None, strides, channels_last=True, ph=ph, use_bias=False
        )
        assert_allclose(deconv_out, linear_out)

        # test dynamic input, explicit dynamic output_shape, NCHW
        ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None,) * (len(y.shape) - 3) + (y_channels, None, None)
        )
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, tf.constant(x_shape), padding,
            kernel, None, strides, channels_last=False, ph=ph, use_bias=False
        )
        assert_allclose(deconv_out, linear_out)

        # if the given payload shape matches the auto-inferred shape
        # further test not giving explicit output_shape
        def axis_matches(i):
            return x_shape[i] == get_deconv_output_length(
                y_shape[i], kernel_size[i], strides[i], padding)
        if all(axis_matches(i) for i in (0, 1)):
            # test static input, implicit output_shape, NHWC
            deconv_out = Deconv2dTestCase.run_deconv2d(
                y, x_channels, kernel_size, None, padding, kernel, None,
                strides, channels_last=True, use_bias=False
            )
            assert_allclose(deconv_out, linear_out)

            # test static input, implicit output_shape, NCHW
            deconv_out = Deconv2dTestCase.run_deconv2d(
                y, x_channels, kernel_size, None, padding, kernel, None,
                strides, channels_last=False, use_bias=False
            )
            assert_allclose(deconv_out, linear_out)

            # test dynamic input, implicit output_shape, NHWC
            ph = tf.placeholder(
                dtype=tf.float32,
                shape=(None,) * (len(y.shape) - 3) + (None, None, y_channels)
            )
            deconv_out = Deconv2dTestCase.run_deconv2d(
                y, x_channels, kernel_size, None, padding, kernel, None,
                strides, channels_last=True, ph=ph,
                use_bias=False
            )
            assert_allclose(deconv_out, linear_out)

            # test dynamic input, implicit output_shape, NCHW
            ph = tf.placeholder(
                dtype=tf.float32,
                shape=(None,) * (len(y.shape) - 3) + (y_channels, None, None)
            )
            deconv_out = Deconv2dTestCase.run_deconv2d(
                y, x_channels, kernel_size, None, padding, kernel, None,
                strides, channels_last=False, ph=ph, use_bias=False
            )
            assert_allclose(deconv_out, linear_out)

        # test normalization and activation
        activation_fn = lambda x: x * 2. + 1.
        normalizer_fn = lambda x: x * 1.5 - 3.
        ans = activation_fn(normalizer_fn(linear_out))

        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, x_shape, padding,
            kernel, bias, strides, channels_last=True,
            normalizer_fn=normalizer_fn, activation_fn=activation_fn
        )
        assert_allclose(deconv_out, ans)

        # test normalization and activation and force using bias
        ans = activation_fn(normalizer_fn(linear_out + bias))
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, x_shape, padding,
            kernel, bias, strides, channels_last=False, use_bias=True,
            normalizer_fn=normalizer_fn, activation_fn=activation_fn
        )
        assert_allclose(deconv_out, ans)

        # test weight norm
        normalized_kernel = l2_normalize(kernel, axis=(0, 1, 2))
        ans = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, x_shape, padding,
            normalized_kernel, None, strides, channels_last=True,
            use_bias=False
        )
        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels, kernel_size, x_shape, padding,
            kernel, None, strides, channels_last=False,
            use_bias=False, weight_norm=True,
            # this can force not using scale in weight_norm
            normalizer_fn=(lambda x: x)
        )
        assert_allclose(deconv_out, ans)

        # test gated
        activation_fn = lambda x: x * 2. + 1.
        normalizer_fn = lambda x: x * 1.5 - 3.
        output, gate = np.split(normalizer_fn(linear_out), 2, axis=-1)
        ans = activation_fn(output) * safe_sigmoid(gate + 1.1)

        deconv_out = Deconv2dTestCase.run_deconv2d(
            y, x_channels // 2, kernel_size, x_shape, padding,
            kernel, bias, strides, channels_last=True,
            normalizer_fn=normalizer_fn, activation_fn=activation_fn,
            gated=True, gate_sigmoid_bias=1.1
        )
        assert_allclose(deconv_out, ans)

    @staticmethod
    def run_deconv2d(input, out_channels, kernel_size, output_shape, padding,
                     kernel, bias, strides, channels_last, ph=None, **kwargs):
        """Run `tfsnippet.layers.conv2d` and get the output."""
        i_shape = input.shape
        if not channels_last:
            input = np.transpose(
                input,
                tuple(i for i in range(len(i_shape) - 3)) + (-1, -3, -2)
            )

        session = tf.get_default_session()
        output = session.run(
            deconv2d(
                input=ph if ph is not None else input,
                out_channels=out_channels,
                kernel_size=kernel_size,
                output_shape=output_shape,
                channels_last=channels_last,
                padding=padding,
                strides=strides,
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

    def test_deconv2d(self):
        with mock.patch('tensorflow.nn.conv2d_transpose',
                        patched_conv2d_transpose), \
                self.test_session() as sess:
            np.random.seed(1234)

            x = np.random.normal(size=[17, 11, 32, 31, 6]).astype(np.float32)
            kernel = np.random.random(size=[3, 4, 6, 7]).astype(np.float32)
            bias = np.random.random(size=[6]).astype(np.float32)

            self.check(x, 'valid', kernel, bias, strides=1)
            self.check(x, 'same', kernel, bias, strides=1)
            self.check(x, 'valid', kernel, bias, strides=(3, 2))
            self.check(x, 'same', kernel, bias, strides=(3, 2))
            # special check: strides == x.shape
            self.check(x, 'valid', kernel, bias, strides=(32, 31))
            self.check(x, 'same', kernel, bias, strides=(32, 31))

    def test_deconv2d_vars(self):
        np.random.seed(1234)

        x = np.random.normal(size=[17, 11, 32, 31, 7]).astype(np.float32)
        kernel = np.random.random(size=[3, 4, 5, 7]).astype(np.float32)
        bias = np.random.random(size=[5]).astype(np.float32)

        # test create variables
        with tf.Graph().as_default():
            # test NHWC
            _ = deconv2d(x, 5, (3, 4), padding='same', channels_last=True)
            assert_variables(['kernel', 'bias'], trainable=True,
                             scope='deconv2d',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

            kernel_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-2]
            bias_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-1]
            self.assertEqual(kernel_var.shape, kernel.shape)
            self.assertEqual(bias_var.shape, bias.shape)

            # test NCHW
            _ = deconv2d(np.transpose(x, [0, 1, -1, -3, -2]), 5, (3, 4),
                         padding='valid', channels_last=False)
            assert_variables(['kernel', 'bias'], trainable=True,
                             scope='deconv2d_1',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

            kernel_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-2]
            bias_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[-1]
            self.assertEqual(kernel_var.shape, kernel.shape)
            self.assertEqual(bias_var.shape, bias.shape)

        # test create variables, non-trainable
        with tf.Graph().as_default():
            # test NHWC
            _ = deconv2d(x, 5, (3, 4), padding='same', channels_last=True,
                         trainable=False)
            assert_variables(['kernel', 'bias'], trainable=False,
                             scope='deconv2d',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test create variables with use_bias = False
        with tf.Graph().as_default():
            _ = deconv2d(x, 5, (3, 4), padding='same', channels_last=True,
                         use_bias=False)
            assert_variables(['kernel'], trainable=True, scope='deconv2d',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])
            assert_variables(['bias'], exist=False, scope='deconv2d')
