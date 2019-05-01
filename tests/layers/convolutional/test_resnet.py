import operator

import numpy as np
import pytest
import tensorflow as tf
from mock import mock, Mock

from tests.layers.core.test_gated import safe_sigmoid
from tfsnippet.layers import *
from tfsnippet.layers.convolutional.resnet import \
    resnet_general_block_apply_gate, resnet_add_shortcut_residual
from tfsnippet.utils import get_static_shape


class TensorFunction(object):

    def __init__(self, tester, fn, input_tag, output_tag, expected_kwargs=()):
        self.tester = tester
        self.fn = fn
        self.input_tag = input_tag
        self.output_tag = output_tag
        self.expected_kwargs = dict(expected_kwargs)

    def __call__(self, input, **kwargs):
        self.tester.assertEqual(input.tag, self.input_tag)
        self.tester.assertDictEqual(kwargs, self.expected_kwargs)
        output = self.fn(input, **kwargs)
        output.tag = self.output_tag
        return output


class TensorOperator(object):

    def __init__(self, tester, fn, input_tag1, input_tag2, output_tag,
                 expected_kwargs=()):
        self.tester = tester
        self.fn = fn
        self.input_tag1 = input_tag1
        self.input_tag2 = input_tag2
        self.output_tag = output_tag
        self.expected_kwargs = dict(expected_kwargs)

    def __call__(self, x, y, **kwargs):
        self.tester.assertEqual(x.tag, self.input_tag1)
        self.tester.assertEqual(y.tag, self.input_tag2)
        self.tester.assertDictEqual(kwargs, self.expected_kwargs)
        output = self.fn(x, y, **kwargs)
        output.tag = self.output_tag
        return output


class ScopeArgTensorFunctionMap(object):

    def __init__(self, scope_map):
        self.scope_map = dict(scope_map)

    def __call__(self, input, scope, **kwargs):
        return self.scope_map[scope](input=input, scope=scope, **kwargs)


class ScopeTensorFunctionMap(object):

    def __init__(self, scope_map):
        self.scope_map = dict(scope_map)

    def __call__(self, *args, **kwargs):
        vs = tf.get_variable_scope().name.rsplit('/')[-1]
        return self.scope_map[vs](*args, **kwargs)


class ResNetBlockTestCase(tf.test.TestCase):

    maxDiff = None

    def test_prerequisites(self):
        t = tf.constant(123.)
        t.new_prop = 456
        t2 = tf.convert_to_tensor(t)
        self.assertIs(t2, t)
        self.assertEqual(t2.new_prop, 456)

    def test_resnet_add_shortcut_residual(self):
        with self.test_session() as sess:
            self.assertEqual(resnet_add_shortcut_residual(1., 2.), 3.)

    def test_general_block_apply_gate(self):
        x = np.random.normal(size=[2, 3, 4, 5, 6]).astype(np.float32)
        y1 = x[..., :3] * safe_sigmoid(x[..., 3:] + 1.1)
        y2 = x[..., :2, :, :] * safe_sigmoid(x[..., 2:, :, :] + 1.1)

        with self.test_session() as sess:
            np.testing.assert_allclose(
                sess.run(resnet_general_block_apply_gate(x, 1.1, axis=-1)),
                y1, rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                sess.run(resnet_general_block_apply_gate(x, 1.1, axis=-3)),
                y2, rtol=1e-5, atol=1e-6
            )

    def test_resnet_general_block(self):
        x = np.random.normal(size=[2, 3, 4, 5, 6]).astype(np.float32)
        x_tensor = tf.convert_to_tensor(x)
        x_tensor.tag = 'input'
        kernel_regularizer = l2_regularizer(0.001)

        # test error arguments
        for arg_name in ('kernel', 'kernel_mask', 'bias'):
            with pytest.raises(ValueError,
                               match='`{}` argument is not allowed for a '
                                     'resnet block'.format(arg_name)):
                _ = resnet_general_block(
                    conv_fn=conv2d,
                    input=x_tensor,
                    in_channels=6,
                    out_channels=8,
                    kernel_size=(3, 2),
                    channels_last=True,
                    **{arg_name: object()}
                )

        # test direct shortcut, without norm, act, dropout
        my_conv2d = Mock(wraps=conv2d)
        conv_fn = ScopeArgTensorFunctionMap({
            'conv_0': TensorFunction(
                self, my_conv2d, 'input', 'conv_0',
                expected_kwargs={
                    'out_channels': 6,
                    'kernel_size': (3, 2),
                    'strides': 1,
                    'channels_last': True,
                    'use_bias': True,
                    'scope': 'conv_0',
                    'kernel_regularizer': kernel_regularizer
                }
            ),
            'conv_1': TensorFunction(
                self, my_conv2d, 'conv_0', 'conv_1',
                expected_kwargs={
                    'out_channels': 6,
                    'kernel_size': (3, 2),
                    'strides': 1,
                    'channels_last': True,
                    'use_bias': True,
                    'scope': 'conv_1',
                    'kernel_regularizer': kernel_regularizer
                }
            )
        })

        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_add_shortcut_residual',
                        TensorOperator(
                            self, operator.add, 'input', 'conv_1', 'output')):
            y = resnet_general_block(
                conv_fn=conv_fn,
                input=x_tensor,
                in_channels=6,
                out_channels=6,
                kernel_size=(3, 2),
                channels_last=True,
                kernel_regularizer=kernel_regularizer
            )
            self.assertEqual(y.tag, 'output')
            self.assertEqual(my_conv2d.call_count, 2)

        # test conv shortcut because of strides != 1, without norm, act, dropout
        my_conv2d = Mock(wraps=conv2d)
        conv_fn = ScopeArgTensorFunctionMap({
            'conv_0': TensorFunction(
                self, my_conv2d, 'input', 'conv_0',
                expected_kwargs={
                    'out_channels': 4,
                    'kernel_size': 1,
                    'strides': 1,
                    'channels_last': False,
                    'use_bias': False,
                    'scope': 'conv_0',
                    'kernel_regularizer': kernel_regularizer
                }
            ),
            'conv_1': TensorFunction(
                self, my_conv2d, 'conv_0', 'conv_1',
                expected_kwargs={
                    'out_channels': 4,
                    'kernel_size': 1,
                    'strides': (2, 2),
                    'channels_last': False,
                    'use_bias': False,
                    'scope': 'conv_1',
                    'kernel_regularizer': kernel_regularizer
                }
            )
        })
        shortcut_conv_fn = TensorFunction(
            self, my_conv2d, 'input', 'shortcut',
            expected_kwargs={
                'out_channels': 4,
                'kernel_size': 1,
                'strides': (2, 2),
                'channels_last': False,
                'use_bias': True,
                'scope': 'shortcut',
                'kernel_regularizer': kernel_regularizer
            }
        )

        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_add_shortcut_residual',
                        TensorOperator(
                            self, operator.add, 'shortcut', 'conv_1',
                            'output'
                        )):
            y = resnet_general_block(
                conv_fn=conv_fn,
                input=x_tensor,
                in_channels=4,
                out_channels=4,
                kernel_size=1,
                strides=(2, 2),
                channels_last=False,
                shortcut_conv_fn=shortcut_conv_fn,
                resize_at_exit=True,
                use_bias=False,
                kernel_regularizer=kernel_regularizer
            )
            self.assertEqual(y.tag, 'output')
            self.assertEqual(my_conv2d.call_count, 3)

        # test conv shortcut because of channel mismatch, w norm, act, dropout
        my_conv2d = Mock(wraps=conv2d)
        tensor_processor = ScopeTensorFunctionMap({
            'norm_0': TensorFunction(
                self, tf.identity, 'input', 'norm_0'),
            'activation_0': TensorFunction(
                self, tf.identity, 'norm_0', 'activation_0'),
            'after_conv_0': TensorFunction(
                self, tf.identity, 'conv_0', 'after_conv_0'),
            'dropout': TensorFunction(
                self, tf.identity, 'after_conv_0', 'dropout'),
            'norm_1': TensorFunction(
                self, tf.identity, 'dropout', 'norm_1'),
            'activation_1': TensorFunction(
                self, tf.identity, 'norm_1', 'activation_1'),
            'after_conv_1': TensorFunction(
                self, tf.identity, 'conv_1', 'after_conv_1'),
        })
        conv_fn = ScopeArgTensorFunctionMap({
            'conv_0': TensorFunction(
                self, my_conv2d, 'activation_0', 'conv_0',
                expected_kwargs={
                    'out_channels': 8,
                    'kernel_size': 2,
                    'strides': 1,
                    'channels_last': True,
                    'use_bias': False,
                    'scope': 'conv_0',
                    'kernel_regularizer': kernel_regularizer
                }
            ),
            'conv_1': TensorFunction(
                self, my_conv2d, 'activation_1', 'conv_1',
                expected_kwargs={
                    'out_channels': 16,
                    'kernel_size': 2,
                    'strides': 1,
                    'channels_last': True,
                    'use_bias': False,
                    'scope': 'conv_1',
                    'kernel_regularizer': kernel_regularizer
                }
            )
        })
        shortcut_conv_fn = TensorFunction(
            self, my_conv2d, 'input', 'shortcut',
            expected_kwargs={
                'out_channels': 8,
                'kernel_size': (3, 2),
                'strides': 1,
                'channels_last': True,
                'use_bias': True,
                'scope': 'shortcut',
                'kernel_regularizer': kernel_regularizer,
            }
        )

        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_general_block_apply_gate',
                        TensorFunction(
                            self, resnet_general_block_apply_gate,
                            'after_conv_1', 'apply_gate',
                            expected_kwargs={
                                'gate_sigmoid_bias': 1.1,
                                'axis': -1,
                            }
                        )), \
                mock.patch('tfsnippet.layers.convolutional.resnet.'
                           'resnet_add_shortcut_residual',
                           TensorOperator(
                               self, operator.add, 'shortcut', 'apply_gate',
                               'output'
                           )):
            y = resnet_general_block(
                conv_fn=conv_fn,
                input=x_tensor,
                in_channels=6,
                out_channels=8,
                kernel_size=2,
                channels_last=True,
                shortcut_conv_fn=shortcut_conv_fn,
                shortcut_kernel_size=(3, 2),
                resize_at_exit=False,
                after_conv_0=tensor_processor,
                after_conv_1=tensor_processor,
                activation_fn=tensor_processor,
                normalizer_fn=tensor_processor,
                dropout_fn=tensor_processor,
                gated=True,
                gate_sigmoid_bias=1.1,
                kernel_regularizer=kernel_regularizer
            )
            self.assertEqual(y.tag, 'output')
            self.assertEqual(y.get_shape()[-1], 8)
            self.assertEqual(my_conv2d.call_count, 3)

        # test conv shortcut because of use_shortcut_conv = True
        my_conv2d = Mock(wraps=conv2d)
        conv_fn = ScopeArgTensorFunctionMap({
            'conv_0': TensorFunction(
                self, my_conv2d, 'input', 'conv_0',
                expected_kwargs={
                    'out_channels': 4,
                    'kernel_size': 1,
                    'strides': 1,
                    'channels_last': False,
                    'use_bias': True,
                    'scope': 'conv_0',
                    'kernel_regularizer': kernel_regularizer
                }
            ),
            'conv_1': TensorFunction(
                self, my_conv2d, 'conv_0', 'conv_1',
                expected_kwargs={
                    'out_channels': 8,
                    'kernel_size': 1,
                    'strides': 1,
                    'channels_last': False,
                    'use_bias': True,
                    'scope': 'conv_1',
                    'kernel_regularizer': kernel_regularizer
                }
            )
        })
        shortcut_conv_fn = TensorFunction(
            self, my_conv2d, 'input', 'shortcut',
            expected_kwargs={
                'out_channels': 4,
                'kernel_size': 1,
                'strides': 1,
                'channels_last': False,
                'use_bias': True,
                'scope': 'shortcut',
                'kernel_regularizer': kernel_regularizer
            }
        )

        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_general_block_apply_gate',
                        TensorFunction(
                            self, resnet_general_block_apply_gate,
                            'conv_1', 'apply_gate',
                            expected_kwargs={
                                'gate_sigmoid_bias': 1.2,
                                'axis': -3,
                            }
                        )), \
                mock.patch('tfsnippet.layers.convolutional.resnet.'
                           'resnet_add_shortcut_residual',
                           TensorOperator(
                               self, operator.add, 'shortcut', 'apply_gate',
                               'output'
                           )):
            y = resnet_general_block(
                conv_fn=conv_fn,
                input=x_tensor,
                in_channels=4,
                out_channels=4,
                kernel_size=1,
                strides=1,
                channels_last=False,
                use_shortcut_conv=True,
                shortcut_conv_fn=shortcut_conv_fn,
                resize_at_exit=True,
                gated=True,
                gate_sigmoid_bias=1.2,
                kernel_regularizer=kernel_regularizer
            )
            self.assertEqual(y.tag, 'output')
            self.assertEqual(my_conv2d.call_count, 3)

    def test_resnet_conv2d_block(self):
        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_general_block',
                        Mock(wraps=resnet_general_block)) as fn:
            normalizer_fn = lambda x: x
            activation_fn = lambda x: x
            dropout_fn = lambda x: x
            after_conv_0 = lambda x: x
            after_conv_1 = lambda x: x
            my_conv2d = lambda *args, **kwargs: conv2d(*args, **kwargs)
            shortcut_conv_fn = lambda *args, **kwargs: conv2d(*args, **kwargs)
            kernel_regularizer = l2_regularizer(0.001)

            # test NHWC
            input = tf.constant(np.random.random(size=[17, 11, 32, 31, 5]),
                                dtype=tf.float32)
            output = resnet_conv2d_block(
                input=input,
                out_channels=7,
                kernel_size=3,
                name='conv_layer',
                kernel_regularizer=kernel_regularizer,
            )
            self.assertEqual(get_static_shape(output), (17, 11, 32, 31, 7))
            self.assertDictEqual(fn.call_args[1], {
                'conv_fn': conv2d,
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': 3,
                'strides': (1, 1),
                'channels_last': True,
                'use_shortcut_conv': None,
                'shortcut_conv_fn': None,
                'shortcut_kernel_size': (1, 1),
                'resize_at_exit': True,
                'after_conv_0': None,
                'after_conv_1': None,
                'activation_fn': None,
                'normalizer_fn': None,
                'dropout_fn': None,
                'gated': False,
                'gate_sigmoid_bias': 2.,
                'use_bias': None,
                'name': 'conv_layer',
                'scope': None,
                'kernel_regularizer': kernel_regularizer,
            })

            # test NCHW
            input = tf.constant(np.random.random(size=[17, 11, 5, 32, 31]),
                                dtype=tf.float32)
            output = resnet_conv2d_block(
                input=input,
                out_channels=7,
                kernel_size=(3, 3),
                conv_fn=my_conv2d,
                strides=2,
                channels_last=False,
                use_shortcut_conv=True,
                shortcut_conv_fn=shortcut_conv_fn,
                shortcut_kernel_size=(2, 2),
                resize_at_exit=False,
                after_conv_0=after_conv_0,
                after_conv_1=after_conv_1,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                dropout_fn=dropout_fn,
                gated=True,
                gate_sigmoid_bias=1.2,
                use_bias=True,
                scope='conv_layer_2',
                kernel_regularizer=kernel_regularizer,
            )
            self.assertEqual(get_static_shape(output), (17, 11, 7, 16, 16))
            self.assertDictEqual(fn.call_args[1], {
                'conv_fn': my_conv2d,
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': (3, 3),
                'strides': 2,
                'channels_last': False,
                'use_shortcut_conv': True,
                'shortcut_conv_fn': shortcut_conv_fn,
                'shortcut_kernel_size': (2, 2),
                'resize_at_exit': False,
                'after_conv_0': after_conv_0,
                'after_conv_1': after_conv_1,
                'activation_fn': activation_fn,
                'normalizer_fn': normalizer_fn,
                'dropout_fn': dropout_fn,
                'gated': True,
                'gate_sigmoid_bias': 1.2,
                'use_bias': True,
                'name': 'resnet_conv2d_block',
                'scope': 'conv_layer_2',
                'kernel_regularizer': kernel_regularizer,
            })

    def test_resnet_deconv2d_block(self):
        with mock.patch('tfsnippet.layers.convolutional.resnet.'
                        'resnet_general_block',
                        Mock(wraps=resnet_general_block)) as fn:
            normalizer_fn = lambda x: x
            activation_fn = lambda x: x
            dropout_fn = lambda x: x
            after_conv_0 = lambda x: x
            after_conv_1 = lambda x: x
            my_deconv2d = Mock(
                wraps=lambda *args, **kwargs: deconv2d(*args, **kwargs))
            shortcut_conv_fn = Mock(
                wraps=lambda *args, **kwargs: deconv2d(*args, **kwargs))
            kernel_regularizer = l2_regularizer(0.001)

            # test NHWC
            input = tf.constant(np.random.random(size=[17, 11, 32, 31, 5]),
                                dtype=tf.float32)
            output = resnet_deconv2d_block(
                input=input,
                out_channels=7,
                kernel_size=3,
                name='deconv_layer',
                kernel_regularizer=kernel_regularizer,
            )
            self.assertEqual(get_static_shape(output), (17, 11, 32, 31, 7))
            kwargs = dict(fn.call_args[1])
            kwargs.pop('conv_fn')
            self.assertDictEqual(kwargs, {
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': 3,
                'strides': (1, 1),
                'channels_last': True,
                'use_shortcut_conv': None,
                'shortcut_conv_fn': None,
                'shortcut_kernel_size': (1, 1),
                'resize_at_exit': False,
                'after_conv_0': None,
                'after_conv_1': None,
                'activation_fn': None,
                'normalizer_fn': None,
                'dropout_fn': None,
                'gated': False,
                'gate_sigmoid_bias': 2.,
                'use_bias': None,
                'name': 'deconv_layer',
                'scope': None,
                'kernel_regularizer': kernel_regularizer,
            })

            # test NCHW
            input = tf.constant(np.random.random(size=[17, 11, 5, 32, 31]),
                                dtype=tf.float32)
            output_shape = (17, 11, 7, 64, 63)
            output = resnet_deconv2d_block(
                input=input,
                out_channels=7,
                kernel_size=(3, 3),
                conv_fn=my_deconv2d,
                strides=2,
                output_shape=output_shape[-2:],
                channels_last=False,
                use_shortcut_conv=True,
                shortcut_conv_fn=shortcut_conv_fn,
                shortcut_kernel_size=(2, 2),
                resize_at_exit=True,
                after_conv_0=after_conv_0,
                after_conv_1=after_conv_1,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                dropout_fn=dropout_fn,
                gated=True,
                gate_sigmoid_bias=1.2,
                use_bias=True,
                scope='deconv_layer_2',
                kernel_regularizer=kernel_regularizer,
            )
            self.assertEqual(get_static_shape(output), output_shape)
            kwargs = dict(fn.call_args[1])
            self.assertIsNot(kwargs.pop('conv_fn'), my_deconv2d)
            self.assertIsNot(kwargs.pop('shortcut_conv_fn'), shortcut_conv_fn)
            self.assertDictEqual(kwargs, {
                'input': input,
                'in_channels': 5,
                'out_channels': 7,
                'kernel_size': (3, 3),
                'strides': 2,
                'channels_last': False,
                'use_shortcut_conv': True,
                'shortcut_kernel_size': (2, 2),
                'resize_at_exit': True,
                'after_conv_0': after_conv_0,
                'after_conv_1': after_conv_1,
                'activation_fn': activation_fn,
                'normalizer_fn': normalizer_fn,
                'dropout_fn': dropout_fn,
                'gated': True,
                'gate_sigmoid_bias': 1.2,
                'use_bias': True,
                'name': 'resnet_deconv2d_block',
                'scope': 'deconv_layer_2',
                'kernel_regularizer': kernel_regularizer,
            })
            self.assertEqual(my_deconv2d.call_count, 2)
            self.assertEqual(shortcut_conv_fn.call_count, 1)
