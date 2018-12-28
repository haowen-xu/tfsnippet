import tensorflow as tf

from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import *
from ..initialization import default_kernel_initializer
from ..utils import validate_weight_norm_arg
from .utils import (validate_conv2d_size_tuple, validate_conv2d_strides_tuple,
                    validate_conv2d_input)

__all__ = ['conv2d']


@add_arg_scope
@add_name_and_scope_arg_doc
def conv2d(input,
           filters,
           kernel_size,
           channels_last=True,
           padding='same',
           strides=(1, 1),
           dilations=1,
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
    """
    2D convolutional layer.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        filters (int): Number of filters (the channel numbers of the output).
        kernel_size (int or (int, int)): Kernel size over spatial dimensions.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        padding: One of {"valid", "same"}, case in-sensitive.
        strides (int or (int, int)): Strides over spatial dimensions.
        dilations (int): The dilation factor over spatial dimensions.
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        weight_norm (bool or (tf.Tensor) -> tf.Tensor)):
            If :obj:`True`, apply :func:`~tfsnippet.layers.weight_norm` on
            `kernel`.  `use_scale` will be :obj:`True` if `normalizer_fn`
            is not specified, and :obj:`False` otherwise.  The axis reduction
            will be determined by the layer.

            If it is a callable function, then it will be used to normalize
            the `kernel` instead of :func:`~tfsnippet.layers.weight_norm`.
            The user must ensure the axis reduction is correct by themselves.
        kernel (Tensor): Instead of creating a new variable, use this tensor.
        kernel_initializer: The initializer for `kernel`.
            Would be ``default_kernel_initializer(...)`` if not specified.
        kernel_regularizer: The regularizer for `kernel`.
        kernel_constraint: The constraint for `kernel`.
        use_bias (bool or None): Whether or not to use `bias`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias (Tensor): Instead of creating a new variable, use this tensor.
        bias_initializer: The initializer for `bias`.
        bias_regularizer: The regularizer for `bias`.
        bias_constraint: The constraint for `bias`.
        trainable (bool): Whether or not the parameters are trainable?

    Returns:
        tf.Tensor: The output tensor.
    """
    input, in_channels, data_format = \
        validate_conv2d_input(input, channels_last)
    dtype = input.dtype.base_dtype

    # check functional arguments
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['VALID', 'SAME'])
    strides = validate_conv2d_strides_tuple('strides', strides, channels_last)
    dilations = validate_positive_int_arg('dilations', dilations)

    if dilations > 1 and not channels_last:
        raise ValueError('`channels_last` == False is incompatible with '
                         '`dilations` > 1.')

    if any(i > 1 for i in strides) and dilations > 1:
        raise ValueError('`strides` > 1 is incompatible with `dilations` > 1.')

    weight_norm_fn = validate_weight_norm_arg(
        weight_norm, axis=-1, use_scale=normalizer_fn is None)
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
        if dilations > 1:
            output = tf.nn.atrous_conv2d(
                value=output,
                filters=kernel,
                rate=dilations,
                padding=padding
            )
        else:
            output = tf.nn.conv2d(
                input=output,
                filter=kernel,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=[1] * 4
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
