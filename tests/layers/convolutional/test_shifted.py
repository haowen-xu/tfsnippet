import functools

import numpy as np
import pytest
import tensorflow as tf
from mock import mock

from tests.layers.convolutional.helper import (input_maybe_to_channels_last,
                                               strides_tuple_to_channels_last,
                                               output_maybe_to_channels_first)
from tfsnippet.layers import shifted_conv2d, conv2d

tf_conv2d = tf.nn.conv2d


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


class ShiftedConv2dTestCase(tf.test.TestCase):

    def test_shifted_conv2d(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)
        x = np.random.normal(size=[3, 11, 13, 7]).astype(np.float32)

        def my_conv2d(*args, **kwargs):
            return conv2d(*args, **kwargs)

        # test the name scope derived by shifted_conv2d
        with tf.Graph().as_default():
            y = shifted_conv2d(
                input=x, out_channels=5, kernel_size=(1, 1),
                spatial_shift=(1, -1), conv_fn=my_conv2d
            )
            self.assertTrue(y.name.startswith('shifted_conv2d/my_conv2d/'))

        # test errors
        with pytest.raises(TypeError,
                           match='`spatial_shift` must be a tuple with two '
                                 'elements, and the elements can only be '
                                 '-1, 0 or 1'):
            _ = shifted_conv2d(input=x, out_channels=5, kernel_size=(2, 3),
                               spatial_shift=(-2, 1))
        with pytest.raises(TypeError,
                           match='`spatial_shift` must be a tuple with two '
                                 'elements, and the elements can only be '
                                 '-1, 0 or 1'):
            _ = shifted_conv2d(input=x, out_channels=5, kernel_size=(2, 3),
                               spatial_shift=(-1,))
        with pytest.raises(ValueError,
                           match='`padding` argument is not supported'):
            _ = shifted_conv2d(input=x, out_channels=5, kernel_size=(2, 3),
                               spatial_shift=(-1, 1), padding='SAME')

        with self.test_session() as sess:
            ####################################################################
            # spatial_shift == (0, 0) should correspond to conv2d SAME padding #
            ####################################################################

            # kernel_size (1, 1)
            kernel = np.random.normal(size=(1, 1, 7, 5)).astype(np.float32)
            assert_allclose(
                *sess.run([
                    shifted_conv2d(
                        input=x, out_channels=5, kernel_size=(1, 1),
                        spatial_shift=(0, 0), kernel=kernel, use_bias=False
                    ),
                    conv2d(
                        input=x, out_channels=5, kernel_size=(1, 1),
                        kernel=kernel, use_bias=False
                    )
                ])
            )

            # kernel_size (2, 3)
            kernel = np.random.normal(size=(2, 3, 7, 5)).astype(np.float32)
            assert_allclose(
                *sess.run([
                    shifted_conv2d(
                        input=x, out_channels=5, kernel_size=(2, 3),
                        spatial_shift=(0, 0), kernel=kernel, use_bias=False
                    ),
                    conv2d(
                        input=x, out_channels=5, kernel_size=(2, 3),
                        kernel=kernel, use_bias=False
                    )
                ])
            )

            ############################
            # spatial_shift == (-1, 0) #
            ############################

            # kernel_size (1, 1), no shift actually
            kernel = np.random.normal(size=(1, 1, 7, 5)).astype(np.float32)
            assert_allclose(
                *sess.run([
                    shifted_conv2d(
                        input=x, out_channels=5, kernel_size=(1, 1),
                        spatial_shift=(-1, 0), kernel=kernel, use_bias=False
                    ),
                    conv2d(
                        input=x, out_channels=5, kernel_size=(1, 1),
                        kernel=kernel, use_bias=False, padding='VALID'
                    )
                ])
            )

            # kernel_size (2, 3), shift accordingly
            kernel = np.random.normal(size=(2, 3, 7, 5)).astype(np.float32)
            x2 = np.zeros([3, 12, 15, 7], dtype=np.float32)
            x2[:, :-1, 1:-1, :] = x
            assert_allclose(
                *sess.run([
                    shifted_conv2d(
                        input=x, out_channels=5, kernel_size=(2, 3),
                        spatial_shift=(-1, 0), kernel=kernel, use_bias=False
                    ),
                    conv2d(
                        input=x2, out_channels=5, kernel_size=(2, 3),
                        kernel=kernel, use_bias=False, padding='VALID'
                    )
                ])
            )

            ############################
            # spatial_shift == (0, 1) #
            ############################

            # kernel_size (1, 1), no shift actually
            kernel = np.random.normal(size=(1, 1, 7, 5)).astype(np.float32)
            assert_allclose(
                *sess.run([
                    shifted_conv2d(
                        input=x, out_channels=5, kernel_size=(1, 1),
                        spatial_shift=(0, 1), kernel=kernel, use_bias=False
                    ),
                    conv2d(
                        input=x, out_channels=5, kernel_size=(1, 1),
                        kernel=kernel, use_bias=False, padding='VALID'
                    )
                ])
            )

            # kernel_size (2, 3), shift accordingly
            kernel = np.random.normal(size=(2, 3, 7, 5)).astype(np.float32)
            x2 = np.zeros([3, 12, 15, 7], dtype=np.float32)
            x2[:, :-1, 2:, :] = x
            assert_allclose(
                *sess.run([
                    shifted_conv2d(
                        input=x, out_channels=5, kernel_size=(2, 3),
                        spatial_shift=(0, 1), kernel=kernel, use_bias=False
                    ),
                    conv2d(
                        input=x2, out_channels=5, kernel_size=(2, 3),
                        kernel=kernel, use_bias=False, padding='VALID'
                    )
                ])
            )

            ##################################
            # spatial_shift == (-1, 1), NCHW #
            ##################################
            x = np.transpose(x, [0, 3, 1, 2])

            with mock.patch('tensorflow.nn.conv2d', patched_conv2d):
                # kernel_size (1, 1), no shift actually
                kernel = np.random.normal(size=(1, 1, 7, 5)).astype(np.float32)
                assert_allclose(
                    *sess.run([
                        shifted_conv2d(
                            input=x, out_channels=5, kernel_size=(1, 1),
                            spatial_shift=(-1, 1), kernel=kernel,
                            use_bias=False, channels_last=False
                        ),
                        conv2d(
                            input=x, out_channels=5, kernel_size=(1, 1),
                            kernel=kernel, use_bias=False,
                            padding='VALID', channels_last=False
                        )
                    ])
                )

                # kernel_size (2, 3), shift accordingly
                kernel = np.random.normal(size=(2, 3, 7, 5)).astype(np.float32)
                x2 = np.zeros([3, 7, 12, 15], dtype=np.float32)
                x2[:, :, :-1, 2:] = x
                assert_allclose(
                    *sess.run([
                        shifted_conv2d(
                            input=x, out_channels=5, kernel_size=(2, 3),
                            spatial_shift=(-1, 1), kernel=kernel,
                            use_bias=False, channels_last=False
                        ),
                        conv2d(
                            input=x2, out_channels=5, kernel_size=(2, 3),
                            kernel=kernel, use_bias=False,
                            padding='VALID', channels_last=False
                        )
                    ])
                )
