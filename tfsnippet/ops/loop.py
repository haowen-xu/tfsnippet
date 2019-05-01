import tensorflow as tf

from tfsnippet.utils import (add_name_arg_doc, is_tensor_object, get_shape,
                             InputSpec, get_static_shape)
from .shape_utils import broadcast_to_shape
from .type_utils import convert_to_tensor_and_cast

__all__ = ['pixelcnn_2d_sample']


@add_name_arg_doc
def pixelcnn_2d_sample(fn, inputs, height, width, channels_last=True,
                       start=0, end=None, back_prop=False,
                       parallel_iterations=1, swap_memory=False,
                       name=None):
    """
    Sample output from a PixelCNN 2D network, pixel-by-pixel.

    Args:
        fn: `(i: tf.Tensor, inputs: tuple[tf.Tensor]) -> tuple[tf.Tensor]`,
            the function to derive the outputs of PixelCNN 2D network at
            iteration `i`.  `inputs` are the pixel-by-pixel outputs gathered
            through iteration `0` to iteration `i - 1`.  The iteration index
            `i` may range from `0` to `height * width - 1`.
        inputs (Iterable[tf.Tensor]): The initial input tensors.
            All the tensors must be at least 4-d, with identical shape.
        height (int or tf.Tensor): The height of the outputs.
        width (int or tf.Tensor): The width of the outputs.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        start (int or tf.Tensor): The start iteration, default `0`.
        end (int or tf.Tensor): The end (exclusive) iteration.
            Default `height * width`.
        back_prop, parallel_iterations, swap_memory: Arguments passed to
            :func:`tf.while_loop`.

    Returns:
        tuple[tf.Tensor]: The final outputs.
    """
    from tfsnippet.layers.convolutional.utils import validate_conv2d_input

    # check the arguments
    def to_int(t):
        if is_tensor_object(t):
            return convert_to_tensor_and_cast(t, dtype=tf.int32)
        return int(t)

    height = to_int(height)
    width = to_int(width)

    inputs = list(inputs)
    if not inputs:
        raise ValueError('`inputs` must not be empty.')
    inputs[0], _, _ = validate_conv2d_input(
        inputs[0], channels_last=channels_last, arg_name='inputs[0]')
    input_spec = InputSpec(shape=get_static_shape(inputs[0]))
    for i, input in enumerate(inputs[1:], 1):
        inputs[i] = input_spec.validate('inputs[{}]'.format(i), input)

    # do pixelcnn sampling
    with tf.name_scope(name, default_name='pixelcnn_2d_sample', values=inputs):
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

        # the input dynamic shape
        input_shape = get_shape(inputs[0])

        # the pixelcnn sampling loop
        def loop_cond(idx, _):
            return idx < end

        def loop_body(idx, inputs):
            inputs = tuple(inputs)

            # prepare for the output mask
            selector = tf.reshape(
                tf.concat(
                    [tf.ones([idx], dtype=tf.uint8),
                     tf.zeros([1], dtype=tf.uint8),
                     tf.ones([total_size - idx - 1], dtype=tf.uint8)],
                    axis=0
                ),
                mask_shape
            )
            selector = tf.cast(broadcast_to_shape(selector, input_shape),
                               dtype=tf.bool)

            # obtain the outputs
            outputs = list(fn(idx, inputs))
            if len(outputs) != len(inputs):
                raise ValueError('The length of outputs != inputs: {} vs {}'.
                                 format(len(outputs), len(inputs)))

            # mask the outputs
            for i, (input, output) in enumerate(zip(inputs, outputs)):
                input_dtype = inputs[i].dtype.base_dtype
                output_dtype = output.dtype.base_dtype
                if output_dtype != input_dtype:
                    raise TypeError(
                        '`outputs[{idx}].dtype` != `inputs[{idx}].dtype`: '
                        '{output} vs {input}'.
                        format(idx=i, output=output_dtype, input=input_dtype)
                    )
                outputs[i] = tf.where(selector, input, output)

            return idx + 1, tuple(outputs)

        i0 = start
        _, outputs = tf.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=(i0, tuple(inputs)),
            back_prop=back_prop,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
        )
        return outputs
