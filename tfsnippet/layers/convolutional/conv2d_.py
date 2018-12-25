import tensorflow as tf

from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import *
from ..initialization import default_kernel_initializer
from ..utils import validate_weight_norm_arg

__all__ = ['conv2d']


def validate_conv2d_size_tuple(arg_name, arg_value):
    """
    Validate the `arg_value`, ensure it is a tuple of two integers, such that
    it can be used as the kernel size, the strides or the dilation rate.

    Args:
        arg_name (str): Name of the argument.
        arg_value: An integer, or a tuple of two integers.
            If it is one integer, it will be duplicated as the two integers.
            Both integers must be positive (>= 1).

    Returns:
        (int, int): The validated two integers.
    """
    arg_value = validate_int_or_int_tuple_arg(arg_name, arg_value)
    if len(arg_value) not in (1, 2) or any(a < 1 for a in arg_value):
        raise ValueError('Invalid value for argument `{}`: expected to be '
                         'one or two positive integers, but got {!r}.'.
                         format(arg_name, arg_value))
    if len(arg_value) == 1:
        arg_value = arg_value * 2
    return arg_value


@add_arg_scope
@add_name_and_scope_arg_doc
def conv2d(input,
           filters,
           kernel_size,
           channels_last=False,
           padding='same',
           strides=(1, 1),
           dilations=(1, 1),
           activation_fn=None,
           normalizer_fn=None,
           weight_norm=False,
           kernel=None,
           kernel_initializer=None,
           kernel_regularizer=None,
           kernel_constraint=None,
           use_bias=None,
           bias=None,
           bias_initializer=tf.zeros_initializer(),
           bias_regularizer=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           scope=None):
    """"""
    # get the specification of inputs
    if channels_last:
        input_spec = InputSpec(shape=('...', '?', '?', '?', '*'))
        channel_axis = -1
        data_format = 'NHWC'
    else:
        input_spec = InputSpec(shape=('...', '?', '*', '?', '?'))
        channel_axis = -3
        data_format = 'NCHW'

    input = input_spec.validate(input)
    dtype = input.dtype.base_dtype
    input_shape = int_shape(input)
    in_channels = input_shape[channel_axis]

    # check functional arguments
    def validate_size_tuple(name, value):
        value = validate_conv2d_size_tuple(name, value)
        if channels_last:
            value = (1,) + value + (1,)
        else:
            value = (1, 1) + value
        return value

    padding = validate_enum_arg(
        'padding', str(padding).lower(), ['valid', 'same'])
    strides = validate_size_tuple('strides', strides)
    dilations = validate_size_tuple('dilations', dilations)

    weight_norm_fn = validate_weight_norm_arg(
        weight_norm, axis=channel_axis, use_scale=normalizer_fn is None)
    if use_bias is None:
        use_bias = normalizer_fn is None

    # get the specification of outputs and parameters
    out_channels = validate_positive_int_arg('filters', filters)
    kernel_size = validate_conv2d_size_tuple('kernel_size', kernel_size)
    kernel_shape = kernel_size + (in_channels, out_channels)
    bias_shape = (out_channels,) if channels_last else (out_channels, 1, 1)

    # validate the parameters
    if kernel is not None:
        kernel = ParamSpec(shape=kernel_shape, dtype=dtype).validate(kernel)
    if kernel_initializer is None:
        kernel_initializer = default_kernel_initializer(weight_norm)
    if bias is not None:
        bias = ParamSpec(shape=bias_shape, dtype=dtype).validate(bias)

    # the main part of the conv2d layer
    with tf.variable_scope(scope, default_name=name or 'conv2d'):
        # create the variables
        if kernel is None:
            kernel = tf.get_variable(
                'kernel',
                shape=kernel_shape,
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
                constraint=kernel_constraint,
                trainable=trainable
            )

        if weight_norm_fn is not None:
            kernel = weight_norm_fn(kernel)

        if use_bias and bias is None:
            bias = tf.get_variable(
                'bias',
                shape=bias_shape,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint,
                trainable=trainable
            )

        # flatten to 4d
        output, s1, s2 = flatten(input, 4)

        # do convolution
        output = tf.nn.conv2d(
            input=output,
            filter=kernel,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilations=dilations
        )

        # add bias
        if use_bias:
            if len(bias_shape) == 1:
                output = tf.nn.bias_add(output, bias)
            else:
                output += bias

        # apply the normalization function if specified
        if normalizer_fn is not None:
            output = normalizer_fn(output)

        # apply the activation function if specified
        if activation_fn is not None:
            output = activation_fn(output)

        # unflatten back to original shape
        output = unflatten(output, s1, s2)

    return output
