from functools import partial

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (add_name_arg_doc, get_static_shape, get_shape,
                             add_name_and_scope_arg_doc, InputSpec)
from tfsnippet.ops import shift
from .utils import validate_conv2d_input
from .conv2d_ import conv2d
from .resnet import resnet_general_block
from .shifted import shifted_conv2d

__all__ = [
    'PixelCNN2DOutput',
    'pixelcnn_2d_input', 'pixelcnn_2d_output',
    'pixelcnn_conv2d_resnet',
]


class PixelCNN2DOutput(object):
    """
    The output of a PixelCNN 2D layer, including tensors from the vertical and
    horizontal convolution stacks.
    """

    def __init__(self, vertical, horizontal):
        """
        Construct a new :class:`PixelCNNLayerOutput`.

        Args:
            vertical (tf.Tensor): The vertical convolution stack output.
            horizontal (tf.Tensor): The horizontal convolution stack output.
        """
        self._vertical = vertical
        self._horizontal = horizontal

    def __repr__(self):
        return 'PixelCNN2DOutput(vertical={},horizontal={})'. \
            format(self.vertical, self.horizontal)

    @property
    def vertical(self):
        """Get the vertical convolution stack output."""
        return self._vertical

    @property
    def horizontal(self):
        """Get the horizontal convolution stack output."""
        return self._horizontal


@add_arg_scope
@add_name_arg_doc
def pixelcnn_2d_input(input, channels_last=True, auxiliary_channel=True,
                      name=None):
    """
    Prepare the input for a PixelCNN 2D network (Tim Salimans, 2017).

    This method must be applied on the input once before any other PixelCNN
    2D layers, for example::

        input = ...  # the input x

        # prepare for the convolution stack
        output = spt.layers.pixelcnn_2d_input(input)

        # apply the PixelCNN 2D layers.
        for i in range(5):
            output = spt.layers.pixelcnn_conv2d_resnet(
                output,
                out_channels=64,
                vertical_kernel_size=(2, 3),
                horizontal_kernel_size=(2, 2),
                activation_fn=tf.nn.leaky_relu,
                normalizer_fn=spt.layers.batch_norm
            )

        # get the final output of the PixelCNN 2D network.
        output = pixelcnn_2d_output(output)

    Args:
        input (Tensor): The input tensor, at least 4-d.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        auxiliary_channel (bool): Whether or not to add a channel to `input`,
            with all elements set to `1`?

    Returns:
        PixelCNN2DOutput: The PixelCNN layer output.
    """
    input, in_channels, _ = validate_conv2d_input(input, channels_last)
    if channels_last:
        h_axis, w_axis, c_axis = -3, -2, -1
    else:
        c_axis, h_axis, w_axis = -3, -2, -1
    rank = len(get_static_shape(input))

    with tf.name_scope(name, default_name='pixelcnn_input', values=[input]):
        # add a channels with all `1`s
        if auxiliary_channel:
            ones_static_shape = [None] * rank
            ones_dynamic_shape = list(ones_static_shape)
            ones_dynamic_shape[c_axis] = 1

            if None in ones_dynamic_shape:
                x_dynamic_shape = get_shape(input)
                for i, s in enumerate(ones_dynamic_shape):
                    if s is None:
                        ones_dynamic_shape[i] = x_dynamic_shape[i]

            ones = tf.ones(shape=tf.stack(ones_dynamic_shape, axis=0),
                           dtype=input.dtype.base_dtype)
            ones.set_shape(tf.TensorShape(ones_static_shape))
            input = tf.concat(
                [input, ones], axis=c_axis, name='auxiliary_input')

        # derive the vertical and horizontal convolution stacks
        down_shift = [0] * rank
        down_shift[h_axis] = 1
        right_shift = [0] * rank
        right_shift[w_axis] = 1

        return PixelCNN2DOutput(
            vertical=shift(input, shift=down_shift, name='vertical'),
            horizontal=shift(input, shift=right_shift, name='horizontal')
        )


@add_name_arg_doc
def pixelcnn_2d_output(input):
    """
    Get the final output of a PixelCNN 2D network from the previous layer.

    Args:
        input (PixelCNN2DOutput): The output from the previous PixelCNN layer.

    Returns:
        tf.Tensor: The output tensor.
    """
    if not isinstance(input, PixelCNN2DOutput):
        raise TypeError('`input` is not an instance of `PixelCNN2DOutput`: '
                        'got {!r}'.format(input))
    return input.horizontal


def pixelcnn_conv2d_resnet_after_conv_0(input, vertical, out_channels,
                                        channels_last, activation_fn,
                                        shortcut_conv_fn, **kwargs):
    if activation_fn is not None:
        with tf.variable_scope('activation'):
            input = activation_fn(input)

    return input + shortcut_conv_fn(
        input=vertical,
        out_channels=out_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        channels_last=channels_last,
        use_bias=True,
        scope='vertical_to_horizontal',
        **kwargs
    )


@add_arg_scope
@add_name_and_scope_arg_doc
def pixelcnn_conv2d_resnet(input,
                           out_channels,
                           conv_fn=conv2d,
                           # the default values for the following two arguments
                           # are from the PixelCNN++ paper.
                           vertical_kernel_size=(2, 3),
                           horizontal_kernel_size=(2, 2),
                           strides=(1, 1),
                           channels_last=True,
                           use_shortcut_conv=None,
                           shortcut_conv_fn=None,
                           shortcut_kernel_size=(1, 1),
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
    PixelCNN 2D convolutional ResNet block.

    Args:
        input (PixelCNN2DOutput): The output from the previous PixelCNN layer.
        out_channels (int): The channel numbers of the output.
        conv_fn: The convolution function for "conv_0" and "conv_1"
            convolutional layers.  See :func:`resnet_general_block`.
        vertical_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for "conv_0" and "conv_1" convolutional layers
            in the PixelCNN vertical stack.
        horizontal_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for "conv_0" and "conv_1" convolutional layers
            in the PixelCNN horizontal stack.
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
        PixelCNN2DOutput: The PixelCNN layer output.
    """
    if not isinstance(input, PixelCNN2DOutput):
        raise TypeError('`input` is not an instance of `PixelCNN2DOutput`: '
                        'got {!r}'.format(input))

    vertical, in_channels, _ = validate_conv2d_input(
        input.vertical, channels_last, 'input.vertical')
    horizontal = InputSpec(shape=get_static_shape(vertical),
                           dtype=vertical.dtype.base_dtype). \
        validate('input.horizontal', input.horizontal)

    if shortcut_conv_fn is None:
        shortcut_conv_fn = conv_fn

    # derive the convolution functions
    vertical_conv_fn = partial(
        shifted_conv2d, conv_fn=conv_fn, spatial_shift=(1, 0))
    horizon_conv_fn = partial(
        shifted_conv2d, conv_fn=conv_fn, spatial_shift=(1, 1))

    with tf.variable_scope(scope,
                           default_name=name or 'pixelcnn_conv2d_resnet'):
        # first, derive the vertical stack output
        vertical = resnet_general_block(
            conv_fn=vertical_conv_fn,
            input=vertical,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=vertical_kernel_size,
            strides=strides,
            channels_last=channels_last,
            use_shortcut_conv=use_shortcut_conv,
            shortcut_conv_fn=shortcut_conv_fn,
            shortcut_kernel_size=shortcut_kernel_size,
            resize_at_exit=False,  # always resize at conv_0
            after_conv_0=None,
            after_conv_1=None,
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            dropout_fn=dropout_fn,
            gated=gated,
            gate_sigmoid_bias=gate_sigmoid_bias,
            use_bias=use_bias,
            scope='vertical',
            **kwargs
        )

        horizontal = resnet_general_block(
            conv_fn=horizon_conv_fn,
            input=horizontal,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=horizontal_kernel_size,
            strides=strides,
            channels_last=channels_last,
            use_shortcut_conv=use_shortcut_conv,
            shortcut_conv_fn=shortcut_conv_fn,
            shortcut_kernel_size=shortcut_kernel_size,
            resize_at_exit=False,  # always resize at conv_0
            after_conv_0=partial(
                pixelcnn_conv2d_resnet_after_conv_0,
                vertical=vertical,
                out_channels=out_channels,
                channels_last=channels_last,
                activation_fn=activation_fn,
                shortcut_conv_fn=shortcut_conv_fn,
                **kwargs
            ),
            after_conv_1=None,
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            dropout_fn=dropout_fn,
            gated=gated,
            gate_sigmoid_bias=gate_sigmoid_bias,
            use_bias=use_bias,
            scope='horizontal',
            **kwargs
        )

    return PixelCNN2DOutput(vertical=vertical, horizontal=horizontal)
