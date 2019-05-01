import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (add_name_and_scope_arg_doc, get_static_shape,
                             get_default_scope_name)
from .conv2d_ import conv2d
from .utils import validate_conv2d_size_tuple, validate_conv2d_input

__all__ = ['shifted_conv2d']


@add_arg_scope
@add_name_and_scope_arg_doc
def shifted_conv2d(input,
                   out_channels,
                   kernel_size,
                   spatial_shift,
                   strides=(1, 1),
                   channels_last=True,
                   conv_fn=conv2d,
                   name=None,
                   scope=None,
                   **kwargs):
    """
    2D convolution with shifted input.

    This method first pads `input` according to the `kernel_size` and
    `spatial_shift` arguments, then do 2D convolution (using `conv_fn`)
    with "VALID" padding.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or (int, int)): Kernel size over spatial dimensions.
        spatial_shift: The `spatial_shift` should be a tuple with two elements
            (corresponding to height and width spatial axes), and the elements
            can only be -1, 0 or 1.

            If the shift for a specific axis is `-1`, then `kernel_size - 1`
            zeros will be padded at the end of that axis.
            If the shift is `0`, then `(kernel_size - 1) // 2` zeros will be
            padded at the front, and `kernel_size // 2` zeros will be padded
            at the end that axis.
            Otherwise if the shift is `1`, then `kernel_size + 1` zeros will
            be padded at the front of that axis.
        strides (int or (int, int)): Strides over spatial dimensions.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        conv_fn: The 2D convolution function. (default :func:`conv2d`)
        \\**kwargs: Other named parameters passed to `conv_fn`.

    Returns:
        tf.Tensor: The output tensor.
    """
    spatial_shift = tuple(spatial_shift)
    if len(spatial_shift) != 2 or \
            any(s not in (-1, 0, 1) for s in spatial_shift):
        raise TypeError('`spatial_shift` must be a tuple with two elements, '
                        'and the elements can only be -1, 0 or 1.')
    kernel_size = validate_conv2d_size_tuple('kernel_size', kernel_size)
    if 'padding' in kwargs:
        raise ValueError('`padding` argument is not supported.')
    input, _, _ = validate_conv2d_input(input, channels_last=channels_last)

    rank = len(get_static_shape(input))
    pads = [(0, 0)] * rank

    is_shifted_conv2d = False
    spatial_start = -3 if channels_last else -2
    for i, (ksize, shift) in enumerate(zip(kernel_size, spatial_shift)):
        axis = i + spatial_start
        if shift == 0:
            pads[axis] = ((ksize - 1) // 2, ksize // 2)
        elif shift == -1:
            pads[axis] = (0, ksize - 1)
            is_shifted_conv2d = True
        else:
            assert(shift == 1)
            pads[axis] = (ksize - 1, 0)
            is_shifted_conv2d = True

    # fast routine: no shift, use ordinary conv_fn with padding == 'SAME'
    if not is_shifted_conv2d:
        return conv_fn(
            input=input,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            channels_last=channels_last,
            padding='SAME',
            scope=scope,
            name=name,
            **kwargs
        )

    # slow routine: pad and use conv_fn with padding == 'VALID'
    with tf.variable_scope(scope, default_name=name or 'shifted_conv2d'):
        output = tf.pad(input, pads)
        output = conv_fn(
            input=output,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            channels_last=channels_last,
            padding='VALID',
            scope=get_default_scope_name(
                getattr(conv_fn, '__name__', None) or 'conv_fn'),
            **kwargs
        )
        return output
