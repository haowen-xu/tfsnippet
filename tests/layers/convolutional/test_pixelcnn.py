import numpy as np
import pytest
import tensorflow as tf
from mock import mock, Mock

from tests.layers.convolutional.test_conv2d import patched_conv2d
from tfsnippet.layers import *
from tfsnippet.layers.convolutional.pixelcnn import \
    pixelcnn_conv2d_resnet_after_conv_0
from tfsnippet.utils import ensure_variables_initialized, get_static_shape


def check_pixel_deps(tester, H, W, vertical, horizontal,
                     vertical_predicate, horizon_predicate,
                     should_print=False):
    for i in range(H):
        for j in range(W):
            idx = i * W + j

            for c in range(vertical.shape[3]):
                # vertical stack influences
                if should_print:
                    print((vertical[idx, ..., c] != 0.).astype(np.int32))

                for h in range(vertical.shape[1]):
                    for w in range(vertical.shape[2]):
                        has_value = (vertical[idx] != 0.)[h, w, c]
                        expect_value = vertical_predicate(i, j, h, w)
                        tester.assertEqual(
                            has_value,
                            expect_value,
                            msg='Vertical stack not match at '
                                '({},{};{},{},{})'.format(i, j, h, w, c)
                        )

                # horizontal stack influences
                if should_print:
                    print((horizontal[idx, ..., c] != 0.).astype(np.int32))

                for h in range(horizontal.shape[1]):
                    for w in range(horizontal.shape[2]):
                        has_value = (horizontal[idx] != 0.)[h, w, c]
                        expect_value = horizon_predicate(i, j, h, w)
                        tester.assertEqual(
                            has_value,
                            expect_value,
                            msg='Horizontal stack not match at '
                                '({},{};{},{},{})'.format(i, j, h, w, c)
                        )


class PixelCNN2DTestCase(tf.test.TestCase):

    def test_pixelcnn_2d_input_and_output(self):
        np.random.seed(1234)

        x = np.random.normal(size=[2, 3, 4, 5, 6]).astype(np.float32)
        x_aug = np.concatenate(
            [x, np.ones([2, 3, 4, 5, 1]).astype(np.float32)],
            axis=-1
        )
        vertical = np.zeros_like(x_aug)
        vertical[..., 1:, :, :] = x_aug[..., :-1, :, :]
        horizontal = np.zeros_like(x_aug)
        horizontal[..., :, 1:, :] = x_aug[..., :, :-1, :]

        with self.test_session() as sess:
            # NHWC
            # auxiliary = True
            output = pixelcnn_2d_input(
                x, channels_last=True, auxiliary_channel=True)
            self.assertIsInstance(output, PixelCNN2DOutput)
            np.testing.assert_allclose(sess.run(output.vertical), vertical)
            np.testing.assert_allclose(sess.run(output.horizontal), horizontal)

            self.assertEqual(
                repr(output),
                'PixelCNN2DOutput(vertical={},horizontal={})'.
                format(output.vertical, output.horizontal)
            )
            self.assertIs(pixelcnn_2d_output(output), output.horizontal)

            # auxiliary = False
            output = pixelcnn_2d_input(
                x, channels_last=True, auxiliary_channel=False)
            self.assertIsInstance(output, PixelCNN2DOutput)
            np.testing.assert_allclose(
                sess.run(output.vertical), vertical[..., :-1])
            np.testing.assert_allclose(
                sess.run(output.horizontal), horizontal[..., :-1])

            # NCHW
            x = np.transpose(x, [0, 1, 4, 2, 3])
            vertical = np.transpose(vertical, [0, 1, 4, 2, 3])
            horizontal = np.transpose(horizontal, [0, 1, 4, 2, 3])

            # auxiliary = True
            output = pixelcnn_2d_input(
                x, channels_last=False, auxiliary_channel=True)
            self.assertIsInstance(output, PixelCNN2DOutput)
            np.testing.assert_allclose(sess.run(output.vertical), vertical)
            np.testing.assert_allclose(sess.run(output.horizontal), horizontal)

            # auxiliary = False
            output = pixelcnn_2d_input(
                x, channels_last=False, auxiliary_channel=False)
            self.assertIsInstance(output, PixelCNN2DOutput)
            np.testing.assert_allclose(
                sess.run(output.vertical), vertical[..., :-1, :, :])
            np.testing.assert_allclose(
                sess.run(output.horizontal), horizontal[..., :-1, :, :])

        with pytest.raises(TypeError, match='`input` is not an instance of '
                                            '`PixelCNN2DOutput`: got 123'):
            _ = pixelcnn_2d_output(123)

    def test_pixelcnn_conv2d_resnet(self):
        with pytest.raises(TypeError, match='`input` is not an instance of '
                                            '`PixelCNN2DOutput`: got 123'):
            _ = pixelcnn_conv2d_resnet(123, out_channels=1)

        with mock.patch('tfsnippet.layers.convolutional.pixelcnn.'
                        'pixelcnn_conv2d_resnet_after_conv_0',
                        Mock(wraps=pixelcnn_conv2d_resnet_after_conv_0)) as m1, \
                mock.patch('tfsnippet.layers.convolutional.pixelcnn.'
                           'resnet_general_block',
                           Mock(wraps=resnet_general_block)) as m2:
            x = pixelcnn_2d_input(
                tf.zeros([2, 6, 8, 5]), auxiliary_channel=False)

            out_channels = 7
            conv_fn = Mock(wraps=conv2d)
            vertical_kernel_size = (3, 4)
            horizontal_kernel_size = (3, 3)
            strides = (2, 1)
            channels_last = False
            use_shortcut_conv = True
            shortcut_conv_fn = Mock(wraps=conv2d)
            shortcut_kernel_size = (1, 2)
            activation_fn = lambda x: x
            normalizer_fn = lambda x: x
            dropout_fn = lambda x: x
            gated = True
            gate_sigmoid_bias = 1.2
            use_bias = True
            kernel_regularizer = l2_regularizer(0.001)

            y = pixelcnn_conv2d_resnet(
                x,
                out_channels=out_channels,
                conv_fn=conv_fn,
                vertical_kernel_size=vertical_kernel_size,
                horizontal_kernel_size=horizontal_kernel_size,
                strides=strides,
                channels_last=channels_last,
                use_shortcut_conv=use_shortcut_conv,
                shortcut_conv_fn=shortcut_conv_fn,
                shortcut_kernel_size=shortcut_kernel_size,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                dropout_fn=dropout_fn,
                gated=gated,
                gate_sigmoid_bias=gate_sigmoid_bias,
                use_bias=use_bias,
                kernel_regularizer=kernel_regularizer
            )
            self.assertIsInstance(y, PixelCNN2DOutput)
            for t in (y.vertical, y.horizontal):
                self.assertTupleEqual(get_static_shape(t),
                                      (2, out_channels, 4, 5))

            self.assertEqual(m1.call_count, 1)
            self.assertEqual(m2.call_count, 2)
            self.assertEqual(conv_fn.call_count, 4)
            self.assertEqual(shortcut_conv_fn.call_count, 3)

            shortcut_kwargs = m1.call_args_list[0][1]
            self.assertEqual(shortcut_kwargs['activation_fn'], activation_fn)
            self.assertEqual(shortcut_kwargs['channels_last'], channels_last)
            self.assertEqual(shortcut_kwargs['out_channels'], out_channels)

            vertical_kwargs = m2.call_args_list[0][1]
            self.assertEqual(vertical_kwargs['input'], x.vertical)
            self.assertEqual(vertical_kwargs['scope'], 'vertical')
            self.assertEqual(vertical_kwargs['kernel_size'],
                             vertical_kernel_size)
            self.assertIsNone(vertical_kwargs['after_conv_0'])

            horizontal_kwargs = m2.call_args_list[1][1]
            self.assertEqual(horizontal_kwargs['input'], x.horizontal)
            self.assertEqual(horizontal_kwargs['scope'], 'horizontal')
            self.assertEqual(horizontal_kwargs['kernel_size'],
                             horizontal_kernel_size)
            self.assertIsNotNone(horizontal_kwargs['after_conv_0'])

            for kwargs in (vertical_kwargs, horizontal_kwargs):
                self.assertEqual(kwargs['in_channels'], 6)
                self.assertEqual(kwargs['out_channels'], out_channels)
                self.assertEqual(kwargs['strides'], strides)
                self.assertEqual(kwargs['channels_last'], channels_last)
                self.assertEqual(kwargs['use_shortcut_conv'], use_shortcut_conv)
                self.assertEqual(kwargs['shortcut_kernel_size'],
                                 shortcut_kernel_size)
                self.assertEqual(kwargs['resize_at_exit'], False)
                self.assertEqual(kwargs['after_conv_1'], None)
                self.assertEqual(kwargs['activation_fn'], activation_fn)
                self.assertEqual(kwargs['normalizer_fn'], normalizer_fn)
                self.assertEqual(kwargs['dropout_fn'], dropout_fn)
                self.assertEqual(kwargs['gated'], gated)
                self.assertEqual(kwargs['gate_sigmoid_bias'], gate_sigmoid_bias)
                self.assertEqual(kwargs['use_bias'], use_bias)
                self.assertEqual(kwargs['kernel_regularizer'],
                                 kernel_regularizer)

    def test_pixelcnn_conv2d_resnet_pixel_dep(self):
        tf.set_random_seed(1234)

        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            ######################
            # test one layer dep #
            ######################
            H, W = 11, 12
            x = np.eye(H * W, dtype=np.float32).reshape([H * W, H, W, 1]) * \
                np.ones([1, 1, 1, 2], dtype=np.float32)

            vertical_predicate = lambda i, j, h, w: (
                (h - 3 <= i <= h - 1) and (w - 2 <= j <= w + 2)
            )
            horizontal_predicate = lambda i, j, h, w: (
                # value from vertical stack
                ((h - 4 <= i <= h - 1) and
                 (w - 3 <= j <= w + 2)) or
                # value from horizontal stack
                ((h - 2 <= i <= h) and
                 (w - 3 <= j <= w - 1))
            )

            # NHWC
            y = pixelcnn_2d_input(
                x, channels_last=True, auxiliary_channel=False)
            o = pixelcnn_conv2d_resnet(
                y, out_channels=3, vertical_kernel_size=(2, 3),
                horizontal_kernel_size=(2, 2), strides=(1, 1),
                channels_last=True,
            )
            ensure_variables_initialized()
            vertical, horizontal = sess.run([o.vertical, o.horizontal])
            check_pixel_deps(
                self, H, W, vertical, horizontal, vertical_predicate,
                horizontal_predicate,
            )

            # NCHW
            y = pixelcnn_2d_input(
                np.transpose(x, [0, 3, 1, 2]), channels_last=False,
                auxiliary_channel=False
            )
            o = pixelcnn_conv2d_resnet(
                y, out_channels=3, vertical_kernel_size=(2, 3),
                horizontal_kernel_size=(2, 2), strides=(1, 1),
                channels_last=False, activation_fn=tf.nn.leaky_relu
            )
            ensure_variables_initialized()
            vertical, horizontal = sess.run([o.vertical, o.horizontal])
            vertical = np.transpose(vertical, [0, 2, 3, 1])
            horizontal = np.transpose(horizontal, [0, 2, 3, 1])
            check_pixel_deps(
                self, H, W, vertical, horizontal, vertical_predicate,
                horizontal_predicate,
            )

            ########################
            # test multi layer dep #
            ########################
            H, W = 6, 7
            x = np.eye(H * W, dtype=np.float32).reshape([H * W, H, W, 1]) * \
                np.ones([1, 1, 1, 2], dtype=np.float32)

            n_layers = 3
            vertical_predicate = lambda i, j, h, w: i <= h - 1
            horizontal_predicate = lambda i, j, h, w: (
                # value from vertical stack
                i <= h - 1 or
                # value from horizontal stack
                (j <= w - 1 and i <= h)
            )

            y = pixelcnn_2d_input(
                x, channels_last=True, auxiliary_channel=False)
            for i in range(n_layers):
                y = pixelcnn_conv2d_resnet(
                    y, out_channels=3, vertical_kernel_size=(2, 3),
                    horizontal_kernel_size=(2, 2), strides=(1, 1),
                    channels_last=True,
                )
            ensure_variables_initialized()
            vertical, horizontal = sess.run([y.vertical, y.horizontal])
            check_pixel_deps(
                self, H, W, vertical, horizontal, vertical_predicate,
                horizontal_predicate,
            )
