import tensorflow as tf

from tfsnippet.utils import add_name_arg_doc, is_tensor_object, get_shape
from .shape_utils import broadcast_to_shape
from .type_utils import convert_to_tensor_and_cast

__all__ = ['pixelcnn_2d_sample']


@add_name_arg_doc
def pixelcnn_2d_sample(fn, x, height, width, channels_last=True,
                       start=0, end=None, name=None):
    """
    Sample output from a PixelCNN 2D network, pixel-by-pixel.

    Args:
        fn: A function ``(i, x) -> y``, where `i` is the iteration index
            (range from `0` to `height * width - 1`), `x` is the output
            obtained pixel-by-pixel through iteration `0` to `i - 1`.
        x (tf.Tensor): The initial `x`, at least 4-d.  Should have exactly
            the same shape as `y` output by `fn`.
        height (int or tf.Tensor): The height of the output.
        width (int or tf.Tensor): The width of the output.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        start (int or tf.Tensor): The start iteration, default `0`.
        end (int or tf.Tensor): The end (exclusive) iteration.
            Default `height * width`.

    Returns:
        tf.Tensor: The final output.
    """
    from tfsnippet.layers.convolutional.utils import validate_conv2d_input

    def to_int(t):
        if is_tensor_object(t):
            return convert_to_tensor_and_cast(t, dtype=tf.int32)
        return int(t)

    height = to_int(height)
    width = to_int(width)
    x, _, _ = validate_conv2d_input(
        x, channels_last=channels_last, arg_name='x')

    with tf.name_scope(name, default_name='pixelcnn_2d_sample', values=[x]):
        # the total size, start and end index
        total_size = height * width
        start = convert_to_tensor_and_cast(start, dtype=tf.int32)
        if end is None:
            end = convert_to_tensor_and_cast(total_size, dtype=tf.int32)
        else:
            end = convert_to_tensor_and_cast(end, dtype=tf.int32)

        # the mask shape
        if channels_last:
            mask_shape = [height, width, 1]
        else:
            mask_shape = [height, width]

        if any(is_tensor_object(t) for t in mask_shape):
            mask_shape = tf.stack(mask_shape, axis=0)

        # the pixelcnn sampling loop
        def loop_cond(i, x):
            return i < end

        def loop_body(i, x):
            dtype = x.dtype.base_dtype
            selector = tf.reshape(
                tf.concat(
                    [tf.ones([i], dtype=tf.uint8),
                     tf.zeros([1], dtype=tf.uint8),
                     tf.ones([total_size - i - 1], dtype=tf.uint8)],
                    axis=0
                ),
                mask_shape
            )
            selector = tf.cast(
                broadcast_to_shape(selector, get_shape(x)), dtype=tf.bool)
            y = convert_to_tensor_and_cast(fn(i, x), dtype=dtype)
            y = tf.where(selector, x, y)
            return i + 1, y

        i0 = start
        _, y = tf.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=[i0, x],
            back_prop=False,
            shape_invariants=[i0.get_shape(), x.get_shape()]
        )
        return y
