from functools import partial

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (validate_int_tuple_arg, is_integer,
                             add_name_and_scope_arg_doc)
from .conv2d_ import conv2d, deconv2d
from .utils import validate_conv2d_input

__all__ = [
    'resnet_general_block',
    'resnet_conv2d_block',
    'resnet_deconv2d_block',
]


def resnet_general_block_apply_gate(input, gate_sigmoid_bias, axis):
    residual, gate = tf.split(input, 2, axis=axis)
    residual = residual * tf.sigmoid(gate + gate_sigmoid_bias, name='gate')
    return residual


def resnet_add_shortcut_residual(x, y):
    return x + y


@add_arg_scope
@add_name_and_scope_arg_doc
def resnet_general_block(conv_fn,
                         input,
                         in_channels,
                         out_channels,
                         kernel_size,
                         strides=1,
                         channels_last=True,
                         use_shortcut_conv=None,
                         shortcut_conv_fn=None,
                         shortcut_kernel_size=1,
                         resize_at_exit=False,
                         after_conv_0=None,
                         after_conv_1=None,
                         activation_fn=None,
                         normalizer_fn=None,
                         dropout_fn=None,
                         gated=False,
                         gate_sigmoid_bias=2.,
                         use_bias=None,
                         name=None,
                         scope=None,
                         **kwargs):
    """
    A general implementation of ResNet block.

    The architecture of this ResNet implementation follows the work
    "Wide residual networks" (Zagoruyko & Komodakis, 2016).  It basically does
    the following things:

    .. code-block:: python

        shortcut = input
        if strides != 1 or in_channels != out_channels or use_shortcut_conv:
            shortcut = shortcut_conv_fn(
                input=shortcut,
                out_channels=out_channels,
                kernel_size=shortcut_kernel_size,
                strides=strides,
                scope='shortcut'
            )

        residual = input
        residual = conv_fn(
            input=activation_fn(normalizer_fn(residual)),
            out_channels=in_channels if resize_at_exit else out_channels,
            kernel_size=kernel_size,
            strides=strides,
            scope='conv_0'
        )
        residual = after_conv_0(residual)
        residual = dropout_fn(residual)
        residual = conv_fn(
            input=activation_fn(normalizer_fn(residual)),
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            scope='conv_1'
        )
        residual = after_conv_1(residual)

        output = shortcut + residual

    Args:
        conv_fn: The convolution function for "conv_0" and "conv_1"
            convolutional layers. It must accept the following named arguments:

            * input
            * out_channels
            * kernel_size
            * strides
            * channels_last
            * use_bias
            * scope

            Also, it must accept the named arguments specified in `kwargs`.
        input (Tensor): The input tensor.
        in_channels (int): The channel numbers of the tensor.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or tuple[int]): Kernel size over spatial dimensions,
            for "conv_0" and "conv_1" convolutional layers.
        strides (int or tuple[int]): Strides over spatial dimensions,
            for "conv_0", "conv_1" and "shortcut" convolutional layers.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        use_shortcut_conv (True or None): If :obj:`True`, force to apply a
            linear convolution transformation on the shortcut path.
            If :obj:`None` (by default), only use shortcut if necessary.
        shortcut_conv_fn: The convolution function for the "shortcut"
            convolutional layer.  It should accept same named arguments
            as `conv_fn`.  If not specified, use `conv_fn`.
        shortcut_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for the "shortcut" convolutional layer.
        resize_at_exit (bool): If :obj:`True`, resize the spatial dimensions
            at the "conv_1" convolutional layer.  If :obj:`False`, resize at
            the "conv_0" convolutional layer. (see above)
        after_conv_0: The function to apply on the output of "conv_0" layer.
        after_conv_1: The function to apply on the output of "conv_1" layer.
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        dropout_fn: The dropout function.
        gated (bool): Whether or not to use gate on the output of "conv_1"?
            `conv_1_output = activation_fn(conv_1_output) * sigmoid(gate)`.
        gate_sigmoid_bias (Tensor): The bias added to `gate` before applying
            the `sigmoid` activation.
        use_bias (bool or None): Whether or not to use `bias` in "conv_0" and
            "conv_1"?  If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        \\**kwargs: Other named arguments passed to "conv_0", "conv_1" and
            "shortcut" convolutional layers.

    Returns:
        tf.Tensor: The output tensor.
    """
    def validate_size_tuple(n, s):
        if is_integer(s):
            # Do not change a single integer into a tuple!
            # This is because we do not know the dimensionality of the
            # convolution operation here, so we cannot build the size
            # tuple with correct number of elements from the integer notation.
            return int(s)
        return validate_int_tuple_arg(n, s)

    def has_non_unit_item(x):
        if is_integer(x):
            return x != 1
        else:
            return any(i != 1 for i in x)

    def apply_fn(fn, x, scope):
        if fn is not None:
            with tf.variable_scope(scope):
                x = fn(x)
        return x

    # check the parameters
    for arg_name in ('kernel', 'kernel_mask', 'bias'):
        if arg_name in kwargs:
            raise ValueError('`{}` argument is not allowed for a resnet block.'.
                             format(arg_name))

    input = tf.convert_to_tensor(input)
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    kernel_size = validate_size_tuple('kernel_size', kernel_size)
    strides = validate_size_tuple('strides', strides)
    shortcut_kernel_size = validate_size_tuple(
        'shortcut_kernel_size', shortcut_kernel_size)

    if use_shortcut_conv is None:
        use_shortcut_conv = \
            has_non_unit_item(strides) or in_channels != out_channels
    if shortcut_conv_fn is None:
        shortcut_conv_fn = conv_fn
    if use_bias is None:
        use_bias = normalizer_fn is None

    # define convolution operations: conv_0, conv_1, and shortcut conv
    def keep_conv(input, n_channels, scope):
        return conv_fn(
            input=input,
            out_channels=n_channels,
            kernel_size=kernel_size,
            strides=1,
            channels_last=channels_last,
            use_bias=use_bias,
            scope=scope,
            **kwargs
        )

    def resize_conv(input, n_channels, scope):
        return conv_fn(
            input=input,
            out_channels=n_channels,
            kernel_size=kernel_size,
            strides=strides,
            channels_last=channels_last,
            use_bias=use_bias,
            scope=scope,
            **kwargs
        )

    n_channels_at_exit = out_channels * 2 if gated else out_channels
    if resize_at_exit:
        conv_0 = partial(
            keep_conv, n_channels=in_channels, scope='conv_0')
        conv_1 = partial(
            resize_conv, n_channels=n_channels_at_exit, scope='conv_1')
    else:
        conv_0 = partial(
            resize_conv, n_channels=out_channels, scope='conv_0')
        conv_1 = partial(
            keep_conv, n_channels=n_channels_at_exit, scope='conv_1')

    if use_shortcut_conv:
        def shortcut_conv(input):
            return shortcut_conv_fn(
                input=input,
                out_channels=out_channels,
                kernel_size=shortcut_kernel_size,
                strides=strides,
                channels_last=channels_last,
                use_bias=True,
                scope='shortcut',
                **kwargs
            )
    else:
        def shortcut_conv(input):
            return input

    with tf.variable_scope(scope, default_name=name or 'resnet_general_block'):
        # build the shortcut path
        shortcut = shortcut_conv(input)

        # build the residual path
        with tf.variable_scope('residual'):
            residual = input
            residual = apply_fn(normalizer_fn, residual, 'norm_0')
            residual = apply_fn(activation_fn, residual, 'activation_0')
            residual = conv_0(residual)
            residual = apply_fn(after_conv_0, residual, 'after_conv_0')
            residual = apply_fn(dropout_fn, residual, 'dropout')
            residual = apply_fn(normalizer_fn, residual, 'norm_1')
            residual = apply_fn(activation_fn, residual, 'activation_1')
            residual = conv_1(residual)
            residual = apply_fn(after_conv_1, residual, 'after_conv_1')
            if gated:
                residual = resnet_general_block_apply_gate(
                    input=residual,
                    gate_sigmoid_bias=gate_sigmoid_bias,
                    axis=-1 if channels_last else -3
                )

        # merge the shortcut path and the residual path
        output = resnet_add_shortcut_residual(shortcut, residual)

    return output


@add_arg_scope
@add_name_and_scope_arg_doc
def resnet_conv2d_block(input,
                        out_channels,
                        kernel_size,
                        conv_fn=conv2d,
                        strides=(1, 1),
                        channels_last=True,
                        use_shortcut_conv=None,
                        shortcut_conv_fn=None,
                        shortcut_kernel_size=(1, 1),
                        resize_at_exit=True,
                        after_conv_0=None,
                        after_conv_1=None,
                        activation_fn=None,
                        normalizer_fn=None,
                        dropout_fn=None,
                        gated=False,
                        gate_sigmoid_bias=2.,
                        use_bias=None,
                        name=None,
                        scope=None,
                        **kwargs):
    """
    2D convolutional ResNet block.

    Args:
        input (Tensor): The input tensor.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or tuple[int]): Kernel size over spatial dimensions,
            for "conv_0" and "conv_1" convolutional layers.
        conv_fn: The convolution function for "conv_0" and "conv_1"
            convolutional layers.  See :func:`resnet_general_block`.
        strides (int or tuple[int]): Strides over spatial dimensions,
            for "conv_0", "conv_1" and "shortcut" convolutional layers.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        use_shortcut_conv (True or None): If :obj:`True`, force to apply a
            linear convolution transformation on the shortcut path.
            If :obj:`None` (by default), only use shortcut if necessary.
        shortcut_conv_fn: The convolution function for the "shortcut"
            convolutional layer.  If not specified, use `conv_fn`.
        shortcut_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for the "shortcut" convolutional layer.
        resize_at_exit (bool): If :obj:`True`, resize the spatial dimensions
            at the "conv_1" convolutional layer.  If :obj:`False`, resize at
            the "conv_0" convolutional layer. (see above)
        after_conv_0: The function to apply on the output of "conv_0" layer.
        after_conv_1: The function to apply on the output of "conv_1" layer.
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        dropout_fn: The dropout function.
        gated (bool): Whether or not to use gate on the output of "conv_1"?
            `conv_1_output = activation_fn(conv_1_output) * sigmoid(gate)`.
        gate_sigmoid_bias (Tensor): The bias added to `gate` before applying
            the `sigmoid` activation.
        use_bias (bool or None): Whether or not to use `bias` in "conv_0" and
            "conv_1"?  If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        \\**kwargs: Other named arguments passed to "conv_0", "conv_1" and
            "shortcut" convolutional layers.

    Returns:
        tf.Tensor: The output tensor.

    See Also:
        :func:`resnet_general_block`
    """
    input, in_channels, _ = validate_conv2d_input(input, channels_last)
    return resnet_general_block(
        conv_fn=conv_fn,
        input=input,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        channels_last=channels_last,
        use_shortcut_conv=use_shortcut_conv,
        shortcut_conv_fn=shortcut_conv_fn,
        shortcut_kernel_size=shortcut_kernel_size,
        resize_at_exit=resize_at_exit,
        after_conv_0=after_conv_0,
        after_conv_1=after_conv_1,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        dropout_fn=dropout_fn,
        gated=gated,
        gate_sigmoid_bias=gate_sigmoid_bias,
        use_bias=use_bias,
        name=name or 'resnet_conv2d_block',
        scope=scope,
        **kwargs
    )


@add_arg_scope
@add_name_and_scope_arg_doc
def resnet_deconv2d_block(input,
                          out_channels,
                          kernel_size,
                          conv_fn=deconv2d,
                          strides=(1, 1),
                          output_shape=None,
                          channels_last=True,
                          use_shortcut_conv=None,
                          shortcut_conv_fn=None,
                          shortcut_kernel_size=(1, 1),
                          resize_at_exit=False,
                          after_conv_0=None,
                          after_conv_1=None,
                          activation_fn=None,
                          normalizer_fn=None,
                          dropout_fn=None,
                          gated=False,
                          gate_sigmoid_bias=2.,
                          use_bias=None,
                          name=None,
                          scope=None,
                          **kwargs):
    """
    2D deconvolutional ResNet block.

    Args:
        input (Tensor): The input tensor.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or tuple[int]): Kernel size over spatial dimensions,
            for "conv_0" and "conv_1" convolutional layers.
        conv_fn: The deconvolution function for "conv_0" and "conv_1"
            deconvolutional layers.  See :func:`resnet_general_block`.
        strides (int or tuple[int]): Strides over spatial dimensions,
            for "conv_0", "conv_1" and "shortcut" deconvolutional layers.
        output_shape: If specified, use this as the shape of the
            deconvolution output; otherwise compute the size of each dimension
            by::

                output_size = input_size * strides
                if padding == 'valid':
                    output_size += max(kernel_size - strides, 0)

        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        use_shortcut_conv (True or None): If :obj:`True`, force to apply a
            linear deconvolution transformation on the shortcut path.
            If :obj:`None` (by default), only use shortcut if necessary.
        shortcut_conv_fn: The deconvolution function for the "shortcut"
            deconvolutional layer.  If not specified, use `conv_fn`.
        shortcut_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for the "shortcut" deconvolutional layer.
        resize_at_exit (bool): If :obj:`True`, resize the spatial dimensions
            at the "conv_1" deconvolutional layer.  If :obj:`False`, resize at
            the "conv_0" deconvolutional layer. (see above)
        after_conv_0: The function to apply on the output of "conv_0" layer.
        after_conv_1: The function to apply on the output of "conv_1" layer.
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        dropout_fn: The dropout function.
        gated (bool): Whether or not to use gate on the output of "conv_1"?
            `conv_1_output = activation_fn(conv_1_output) * sigmoid(gate)`.
        gate_sigmoid_bias (Tensor): The bias added to `gate` before applying
            the `sigmoid` activation.
        use_bias (bool or None): Whether or not to use `bias` in "conv_0" and
            "conv_1"?  If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        \\**kwargs: Other named arguments passed to "conv_0", "conv_1" and
            "shortcut" deconvolutional layers.

    Returns:
        tf.Tensor: The output tensor.

    See Also:
        :func:`resnet_general_block`
    """
    input, in_channels, _ = validate_conv2d_input(input, channels_last)

    def add_output_shape_arg(conv_fn):
        def wrapper(strides, **kwargs):
            if strides == 1:
                return conv_fn(strides=strides, **kwargs)
            else:
                return conv_fn(strides=strides, output_shape=output_shape,
                               **kwargs)
        return wrapper

    conv_fn = add_output_shape_arg(conv_fn)
    if shortcut_conv_fn is not None:
        shortcut_conv_fn = add_output_shape_arg(shortcut_conv_fn)

    # build the resnet block
    return resnet_general_block(
        conv_fn=conv_fn,
        input=input,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        channels_last=channels_last,
        use_shortcut_conv=use_shortcut_conv,
        shortcut_conv_fn=shortcut_conv_fn,
        shortcut_kernel_size=shortcut_kernel_size,
        resize_at_exit=resize_at_exit,
        after_conv_0=after_conv_0,
        after_conv_1=after_conv_1,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        dropout_fn=dropout_fn,
        gated=gated,
        gate_sigmoid_bias=gate_sigmoid_bias,
        use_bias=use_bias,
        name=name or 'resnet_deconv2d_block',
        scope=scope,
        **kwargs
    )
