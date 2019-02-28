import numpy as np
import pytest
import tensorflow as tf
from mock import mock

from tests.layers.convolutional.test_conv2d import patched_conv2d
from tfsnippet.layers import *
from tfsnippet.utils import ensure_variables_initialized


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

    def test_pixelcnn_conv2d_resnet_deps(self):
        tf.set_random_seed(1234)

        H, W = 11, 12
        x = np.eye(H * W, dtype=np.float32).reshape([H * W, H, W, 1])

        def check_influence(vertical, horizontal):
            for i in range(H):
                for j in range(W):
                    idx = i * W + j

                    # vertical stack influences
                    for h in range(H):
                        for w in range(W):
                            has_value = (vertical[idx] != 0.)[h, w]
                            expect_value = ((h - 3 <= i <= h - 1) and
                                            (w - 2 <= j <= w + 2))
                            self.assertEqual(
                                has_value,
                                expect_value,
                                msg='Vertical stack not match at '
                                    '({},{},{},{})'.format(i, j, h, w)
                            )

                    # horizontal stack influences
                    for h in range(H):
                        for w in range(W):
                            has_value = (horizontal[idx] != 0.)[h, w]
                            expect_value = (
                                # value from vertical stack
                                ((h - 4 <= i <= h - 1) and
                                 (w - 3 <= j <= w + 2)) or
                                # value from horizontal stack
                                ((h - 2 <= i <= h) and
                                 (w - 3 <= j <= w - 1))
                            )
                            self.assertEqual(
                                has_value,
                                expect_value,
                                msg='Horizontal stack not match at '
                                    '({},{},{},{})'.format(i, j, h, w)
                            )

        with mock.patch('tensorflow.nn.conv2d', patched_conv2d), \
                self.test_session() as sess:
            # NHWC
            y = pixelcnn_2d_input(
                x, channels_last=True, auxiliary_channel=False)
            o = pixelcnn_conv2d_resnet(
                y, out_channels=1, vertical_kernel_size=(2, 3),
                horizontal_kernel_size=(2, 2), strides=(1, 1),
                channels_last=True,
            )
            ensure_variables_initialized()
            vertical, horizontal = sess.run([o.vertical, o.horizontal])
            check_influence(
                vertical.reshape(vertical.shape[:-1]),
                horizontal.reshape(horizontal.shape[:-1])
            )

            # NCHW
            y = pixelcnn_2d_input(
                np.transpose(x, [0, 3, 1, 2]), channels_last=False,
                auxiliary_channel=False
            )
            o = pixelcnn_conv2d_resnet(
                y, out_channels=1, vertical_kernel_size=(2, 3),
                horizontal_kernel_size=(2, 2), strides=(1, 1),
                channels_last=False, activation_fn=tf.nn.leaky_relu
            )
            ensure_variables_initialized()
            vertical, horizontal = sess.run([o.vertical, o.horizontal])
            vertical = np.transpose(vertical, [0, 2, 3, 1])
            horizontal = np.transpose(horizontal, [0, 2, 3, 1])
            check_influence(
                vertical.reshape(vertical.shape[:-1]),
                horizontal.reshape(horizontal.shape[:-1])
            )

        with pytest.raises(TypeError, match='`input` is not an instance of '
                                            '`PixelCNN2DOutput`: got 123'):
            _ = pixelcnn_conv2d_resnet(123, out_channels=1)
