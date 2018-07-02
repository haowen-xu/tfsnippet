import functools

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.examples.utils import (int_shape,
                                      validate_strides_or_filter_arg,
                                      add_variable_scope)

__all__ = [
    'global_average_pooling',
    'resnet_block',
    'deconv_resnet_block',
    'reshape_conv2d_to_flat',
]


@add_arg_scope
@add_variable_scope
def global_average_pooling(inputs, channels_last=False, name=None):
    """
    Global average pooling layer.

    Args:
        inputs: The inputs feature map.
        channels_last: Whether or not the channels are the last dimension?
            (default :obj:`False`)
        name: Optional name of the variable scope.

    Returns:
        The output feature map.
    """
    if channels_last:
        kernel_size = int_shape(inputs)[1: 3]
        data_format = 'channels_last'
    else:
        kernel_size = int_shape(inputs)[2: 4]
        data_format = 'channels_first'
    return tf.layers.average_pooling2d(inputs, kernel_size, (1, 1),
                                       padding='valid', data_format=data_format)


def _resnet_block(conv_fn, inputs, input_shape, output_shape,
                  kernel_size, strides, channels_last, resize_last,
                  activation_fn, normalizer_fn, dropout_fn):
    # check the arguments
    kernel_size = validate_strides_or_filter_arg('filters', kernel_size)
    strides = validate_strides_or_filter_arg('strides', strides)
    data_format = 'channels_last' if channels_last else 'channels_first'

    # normalization and activation functions
    def add_scope(method):
        @six.wraps(method)
        def wrapper(inputs, name):
            with tf.name_scope(name):
                return method(inputs)
        return wrapper

    activation_fn = add_scope(activation_fn or (lambda x: x))
    normalizer_fn = add_scope(normalizer_fn or (lambda x: x))
    dropout_fn = add_scope(dropout_fn or (lambda x: x))

    # convolutional functions
    resize_conv = lambda inputs, kernel_size, name: conv_fn(
        inputs, output_shape, kernel_size=kernel_size, strides=strides,
        padding='same', data_format=data_format, name=name
    )
    keep_conv = lambda inputs, kernel_size, name: conv_fn(
        inputs, input_shape, kernel_size=kernel_size, strides=(1, 1),
        padding='same', data_format=data_format, name=name
    )

    # build the shortcut path
    if strides != (1, 1):
        shortcut = resize_conv(inputs, (1, 1), 'shortcut')
    else:
        shortcut = inputs

    # build the residual path
    if resize_last:
        conv1, conv2 = keep_conv, resize_conv
    else:
        conv1, conv2 = resize_conv, keep_conv
    residual = inputs
    residual = normalizer_fn(residual, 'norm1')
    residual = activation_fn(residual, 'nonlinear1')
    residual = conv1(residual, kernel_size, 'conv1')
    residual = dropout_fn(residual, 'dropout1')
    residual = normalizer_fn(residual, 'norm2')
    residual = activation_fn(residual, 'nonlinear2')
    residual = conv2(residual, kernel_size, 'conv2')

    # final do the merge
    return shortcut + residual


def _partial_conv(conv_fn,
                  normalizer_fn=None,
                  kernel_initializer=None,
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None):
    return functools.partial(
        conv_fn,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_bias=normalizer_fn is None
    )


@add_arg_scope
@add_variable_scope
def resnet_block(inputs,
                 output_dims,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 channels_last=False,
                 activation_fn=None,
                 normalizer_fn=None,
                 dropout_fn=None,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name=None):
    inputs = tf.convert_to_tensor(inputs)
    input_shape = int(inputs.get_shape()[3 if channels_last else 1])
    output_shape = int(output_dims)
    return _resnet_block(
        conv_fn=_partial_conv(
            tf.layers.conv2d,
            normalizer_fn=normalizer_fn,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        ),
        inputs=inputs,
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size=kernel_size,
        strides=strides,
        channels_last=channels_last,
        resize_last=True,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        dropout_fn=dropout_fn,
    )


@add_arg_scope
@add_variable_scope
def deconv_resnet_block(inputs,
                        output_dims,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        channels_last=False,
                        activation_fn=None,
                        normalizer_fn=None,
                        dropout_fn=None,
                        kernel_initializer=None,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        name=None):
    inputs = tf.convert_to_tensor(inputs)
    input_shape = int(inputs.get_shape()[3 if channels_last else 1])
    output_shape = int(output_dims)
    return _resnet_block(
        conv_fn=_partial_conv(
            tf.layers.conv2d_transpose,
            normalizer_fn=normalizer_fn,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        ),
        inputs=inputs,
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size=kernel_size,
        strides=strides,
        channels_last=channels_last,
        resize_last=True,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        dropout_fn=dropout_fn,
    )


def reshape_conv2d_to_flat(inputs, name=None):
    """
    Reshape the 2-d convolutional output `inputs` to flat vector.

    Args:
        inputs: 4-d Tensor, 2-d convolutional output.
        name (None or str): Name of this operation.

    Returns:
        tf.Tensor: 2-d flatten vector.
    """
    with tf.name_scope(name, default_name='reshape_conv_to_flat',
                       values=[inputs]):
        out_shape = tf.concat(
            [tf.shape(inputs)[:-3], [np.prod(int_shape(inputs)[-3:])]],
            axis=0
        )
        return tf.reshape(inputs, out_shape)
