import tensorflow as tf

from tfsnippet.utils import (get_static_shape, get_shape, add_name_arg_doc,
                             is_tensor_object, assert_deps)

__all__ = ['shift']


@add_name_arg_doc
def shift(input, shift, name=None):
    """
    Shift each axis of `input` according to `shift`, but keep identical size.
    The extra content will be discarded if shifted outside the original size.
    Zeros will be padded to the front or end of shifted axes.

    Args:
        input (Tensor): The tensor to be shifted.
        shift (Iterable[int]): The shift length for each axes.
            It must be equal to the rank of `input`.
            For each axis, if its corresponding shift < 0, then the
            `input` will be shifted to left by `-shift` at that axis.
            If its shift > 0, then the `input` will be shifted to right
            by `shift` at that axis.

    Returns:
        tf.Tensor: The output tensor.
    """
    shift = tuple(int(s) for s in shift)
    input = tf.convert_to_tensor(input)
    shape = get_static_shape(input)

    if shape is None:
        raise ValueError('The rank of `shape` is required to be deterministic: '
                         'got {}'.format(input))
    if len(shift) != len(shape):
        raise ValueError('The length of `shift` is required to equal the rank '
                         'of `input`: shift {} vs input {}'.
                         format(shift, input))

    # cache for the dynamic shape
    def get_dynamic_shape():
        if cached[0] is None:
            cached[0] = get_shape(input)
        return cached[0]
    cached = [None]

    # main routine
    with tf.name_scope(name, default_name='shift', values=[input]):
        # compute the slicing and padding arguments
        has_shift = False
        assert_ops = []
        slice_begin = []
        slice_size = []
        paddings = []
        err_msg = ('Cannot shift `input`: input {} vs shift {}'.
                   format(input, shift))

        for i, (axis_shift, axis_size) in enumerate(zip(shift, shape)):
            # fast approach: shift is zero, no slicing at the axis
            if axis_shift == 0:
                slice_begin.append(0)
                slice_size.append(-1)
                paddings.append((0, 0))
                continue

            # slow approach: shift is not zero, should slice at the axis
            axis_shift_abs = abs(axis_shift)

            # we first check whether or not the axis size is big enough
            if axis_size is None:
                dynamic_axis_size = get_dynamic_shape()[i]
                assert_ops.append(
                    tf.assert_greater_equal(
                        dynamic_axis_size, axis_shift_abs, message=err_msg))
            else:
                if axis_size < axis_shift_abs:
                    raise ValueError(err_msg)

            # next, we compose the slicing range
            if axis_shift < 0:  # shift to left
                slice_begin.append(-axis_shift)
                slice_size.append(-1)
                paddings.append((0, -axis_shift))

            else:  # shift to right
                slice_begin.append(0)
                if axis_size is None:
                    slice_size.append(get_dynamic_shape()[i] - axis_shift)
                else:
                    slice_size.append(axis_size - axis_shift)
                paddings.append((axis_shift, 0))

            # mark the flag to indicate that we've got any axis to shift
            has_shift = True

        if assert_ops:
            with assert_deps(assert_ops) as asserted:
                if asserted:
                    input = tf.identity(input)

        # no axis to shift, directly return the input
        if not has_shift:
            return input

        # do slicing and padding
        if any(is_tensor_object(s) for s in slice_size):
            slice_size = tf.stack(slice_size, axis=0)

        output = tf.slice(input, slice_begin, slice_size)
        output = tf.pad(output, paddings)

        return output
