import numpy as np
import tensorflow as tf

from tfsnippet.utils import (add_name_arg_doc, get_static_shape, concat_shapes,
                             get_shape, is_tensor_object, assert_deps,
                             InputSpec)
from .control_flows import smart_cond
from .assertions import assert_rank, assert_rank_at_least

__all__ = [
    'prepend_dims',
    'flatten_to_ndims',
    'unflatten_from_ndims',
    'broadcast_to_shape',
    'broadcast_to_shape_strict',
    'broadcast_concat',
    'transpose_conv2d_axis',
    'transpose_conv2d_channels_last_to_x',
    'transpose_conv2d_channels_x_to_last',
    'reshape_tail',
]


@add_name_arg_doc
def prepend_dims(x, ndims=1, name=None):
    """
    Prepend `[1] * ndims` to the beginning of the shape of `x`.

    Args:
        x: The tensor `x`.
        ndims: Number of `1` to prepend.

    Returns:
        tf.Tensor: The tensor with prepended dimensions.
    """
    ndims = int(ndims)
    if ndims < 0:
        raise ValueError('`ndims` must be >= 0: got {}'.format(ndims))

    x = tf.convert_to_tensor(x)
    if ndims == 0:
        return x

    with tf.name_scope(name, default_name='prepend_dims', values=[x]):
        static_shape = get_static_shape(x)
        if static_shape is not None:
            static_shape = tf.TensorShape([1] * ndims + list(static_shape))

        dynamic_shape = concat_shapes([
            [1] * ndims,
            get_shape(x)
        ])

        y = tf.reshape(x, dynamic_shape)
        if static_shape is not None:
            y.set_shape(static_shape)

        return y


@add_name_arg_doc
def flatten_to_ndims(x, ndims, name=None):
    """
    Flatten the front dimensions of `x`, such that the resulting tensor
    will have at most `ndims` dimensions.

    Args:
        x (Tensor): The tensor to be flatten.
        ndims (int): The maximum number of dimensions for the resulting tensor.

    Returns:
        (tf.Tensor, tuple[int or None], tuple[int] or tf.Tensor) or (tf.Tensor, None, None):
            (The flatten tensor, the static front shape, and the front shape),
            or (the original tensor, None, None)
    """
    x = tf.convert_to_tensor(x)
    if ndims < 1:
        raise ValueError('`k` must be greater or equal to 1.')
    if not x.get_shape():
        raise ValueError('`x` is required to have known number of '
                         'dimensions.')
    shape = get_static_shape(x)
    if len(shape) < ndims:
        raise ValueError('`k` is {}, but `x` only has rank {}.'.
                         format(ndims, len(shape)))
    if len(shape) == ndims:
        return x, None, None

    with tf.name_scope(name, default_name='flatten', values=[x]):
        if ndims == 1:
            static_shape = shape
            if None in shape:
                shape = tf.shape(x)
            return tf.reshape(x, [-1]), static_shape, shape
        else:
            front_shape, back_shape = shape[:-(ndims - 1)], shape[-(ndims - 1):]
            static_front_shape = front_shape
            static_back_shape = back_shape
            if None in front_shape or None in back_shape:
                dynamic_shape = tf.shape(x)
                if None in front_shape:
                    front_shape = dynamic_shape[:-(ndims - 1)]
                if None in back_shape:
                    back_shape = dynamic_shape[-(ndims - 1):]
            if isinstance(back_shape, tuple):
                x = tf.reshape(x, [-1] + list(back_shape))
            else:
                x = tf.reshape(x, tf.concat([[-1], back_shape], axis=0))
                x.set_shape(tf.TensorShape([None] + list(static_back_shape)))
            return x, static_front_shape, front_shape


@add_name_arg_doc
def unflatten_from_ndims(x, static_front_shape, front_shape, name=None):
    """
    The inverse transformation of :func:`flatten`.

    If both `static_front_shape` is None and `front_shape` is None,
    `x` will be returned without any change.

    Args:
        x (Tensor): The tensor to be unflatten.
        static_front_shape (tuple[int or None] or None): The static front shape.
        front_shape (tuple[int] or tf.Tensor or None): The front shape.

    Returns:
        tf.Tensor: The unflatten x.
    """
    x = tf.convert_to_tensor(x)
    if static_front_shape is None and front_shape is None:
        return x
    if not x.get_shape():
        raise ValueError('`x` is required to have known number of '
                         'dimensions.')
    shape = get_static_shape(x)
    if len(shape) < 1:
        raise ValueError('`x` only has rank {}, required at least 1.'.
                         format(len(shape)))
    if not is_tensor_object(front_shape):
        front_shape = tuple(front_shape)

    with tf.name_scope(name, default_name='unflatten', values=[x]):
        back_shape = shape[1:]
        static_back_shape = back_shape
        if None in back_shape:
            back_shape = tf.shape(x)[1:]
        if isinstance(front_shape, tuple) and isinstance(back_shape, tuple):
            x = tf.reshape(x, front_shape + back_shape)
        else:
            x = tf.reshape(x, tf.concat([front_shape, back_shape], axis=0))
            x.set_shape(tf.TensorShape(list(static_front_shape) +
                                       list(static_back_shape)))
        return x


def broadcast_to_shape(x, shape, name=None):
    """
    Broadcast `x` to match `shape`.

    If ``rank(x) > len(shape)``, only the tail dimensions will be broadcasted
    to match `shape`.

    Args:
        x: A tensor.
        shape (tuple[int] or tf.Tensor): Broadcast `x` to match this shape.

    Returns:
        tf.Tensor: The broadcasted tensor.
    """
    # check the parameters
    x = tf.convert_to_tensor(x)
    x_shape = get_static_shape(x)
    ns_values = [x]
    if is_tensor_object(shape):
        shape = tf.convert_to_tensor(shape)
        ns_values.append(shape)
    else:
        shape = tuple(int(s) for s in shape)

    with tf.name_scope(name=name or 'broadcast_to_shape', values=ns_values):
        cannot_broadcast_msg = (
            '`x` cannot be broadcasted to match `shape`: x {!r} vs shape {!r}'.
            format(x, shape)
        )

        # fast routine: shape is tuple[int] and x_shape is all known,
        # we can use reshape + tile to do the broadcast, which should be faster
        # than using ``x * ones(shape)``.
        if isinstance(shape, tuple) and x_shape is not None and \
                all(s is not None for s in x_shape):
            # reshape to have the same dimension
            if len(x_shape) < len(shape):
                x_shape = (1,) * (len(shape) - len(x_shape)) + x_shape
                x = tf.reshape(x, x_shape)

            # tile to have the same shape
            tile = []
            i = -1
            while i > -len(shape) - 1:
                a, b = x_shape[i], shape[i]
                if a == 1 and b > 1:
                    tile.append(b)
                elif a != b:
                    raise ValueError(cannot_broadcast_msg)
                else:
                    tile.append(1)
                i -= 1
            tile = [1] * (len(x_shape) - len(shape)) + list(reversed(tile))
            if any(s > 1 for s in tile):
                x = tf.tile(x, tile)

            return x

        # slow routine: we may need ``x * ones(shape)`` to do the broadcast
        assertions = []
        post_assert_shape = False
        static_shape = tf.TensorShape(None)

        if isinstance(shape, tuple) and x_shape is not None:
            need_multiply_ones = False

            # it should always broadcast if len(x_shape) < len(shape)
            if len(x_shape) < len(shape):
                need_multiply_ones = True

            # check the consistency of x and shape
            static_shape_hint = []  # list to gather the static shape hint
            axis_to_check = []  # list to gather the axis to check
            i = -1
            while i >= -len(shape) and i >= -len(x_shape):
                a, b = x_shape[i], shape[i]
                if a is None:
                    axis_to_check.append(i)
                else:
                    if a != b:
                        if a == 1:
                            need_multiply_ones = True
                        else:
                            raise ValueError(cannot_broadcast_msg)
                static_shape_hint.append(b)
                i -= 1

            # compose the static shape hint
            if len(shape) < len(x_shape):
                static_shape = x_shape[:-len(shape)]
            elif len(shape) > len(x_shape):
                static_shape = shape[:-len(x_shape)]
            else:
                static_shape = ()
            static_shape = tf.TensorShape(
                static_shape + tuple(reversed(static_shape_hint)))

            # compose the assertion operations and the multiply flag
            if axis_to_check:
                need_multiply_flags = []
                x_dynamic_shape = tf.shape(x)

                for i in axis_to_check:
                    assertions.append(tf.assert_equal(
                        tf.logical_or(
                            tf.equal(x_dynamic_shape[i], shape[i]),
                            tf.equal(x_dynamic_shape[i], 1),
                        ),
                        True,
                        message=cannot_broadcast_msg
                    ))
                    if len(x_shape) >= len(shape):
                        need_multiply_flags.append(
                            tf.not_equal(x_dynamic_shape[i], shape[i]))

                if not need_multiply_ones:
                    need_multiply_ones = \
                        tf.reduce_any(tf.stack(need_multiply_flags))

        else:
            # we have no ideal about what `shape` is here, thus we need to
            # assert the shape after ``x * ones(shape)``.
            need_multiply_ones = True
            post_assert_shape = True

        # do broadcast if `x_shape` != `shape`
        def multiply_branch():
            with assert_deps(assertions):
                ones_template = tf.ones(shape, dtype=x.dtype.base_dtype)
            try:
                return x * ones_template
            except ValueError:  # pragma: no cover
                raise ValueError(cannot_broadcast_msg)

        def identity_branch():
            with assert_deps(assertions) as asserted:
                if asserted:
                    return tf.identity(x)
                else:  # pragma: no cover
                    return x

        t = smart_cond(need_multiply_ones, multiply_branch, identity_branch)
        t.set_shape(static_shape)

        if post_assert_shape:
            post_assert_op = tf.assert_equal(
                tf.reduce_all(tf.equal(tf.shape(t)[-tf.size(shape):], shape)),
                True,
                message=cannot_broadcast_msg
            )
            with assert_deps([post_assert_op]) as asserted:
                if asserted:
                    t = tf.identity(t)

        return t


@add_name_arg_doc
def broadcast_to_shape_strict(x, shape, name=None):
    """
    Broadcast `x` to match `shape`.

    This method requires `rank(x)` to be less than or equal to `len(shape)`.
    You may use :func:`broadcast_to_shape` instead, to allow the cases where
    ``rank(x) > len(shape)``.

    Args:
        x: A tensor.
        shape (tuple[int] or tf.Tensor): Broadcast `x` to match this shape.

    Returns:
        tf.Tensor: The broadcasted tensor.
    """
    # check the parameters
    x = tf.convert_to_tensor(x)
    x_shape = get_static_shape(x)
    ns_values = [x]
    if is_tensor_object(shape):
        shape = tf.convert_to_tensor(shape)
        ns_values.append(shape)
    else:
        shape = tuple(int(s) for s in shape)

    with tf.name_scope(name=name or 'broadcast_to_shape', values=ns_values):
        cannot_broadcast_msg = (
            '`x` cannot be broadcasted to match `shape`: x {!r} vs shape {!r}'.
            format(x, shape)
        )

        # assert ``rank(x) <= len(shape)``
        if isinstance(shape, tuple) and x_shape is not None:
            if len(x_shape) > len(shape):
                raise ValueError(cannot_broadcast_msg)
        elif isinstance(shape, tuple):
            with assert_deps([
                        tf.assert_less_equal(
                            tf.rank(x),
                            len(shape),
                            message=cannot_broadcast_msg
                        )
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    x = tf.identity(x)
        else:
            with assert_deps([
                        assert_rank(
                            shape,
                            1,
                            message=cannot_broadcast_msg
                        )
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    shape = tf.identity(shape)

            with assert_deps([
                        tf.assert_less_equal(
                            tf.rank(x),
                            tf.size(shape),
                            message=cannot_broadcast_msg
                        )
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    x = tf.identity(x)

        # do broadcast
        return broadcast_to_shape(x, shape)


@add_name_arg_doc
def broadcast_concat(x, y, axis, name=None):
    """
    Broadcast `x` and `y`, then concat them along `axis`.

    This method cannot deal with all possible situations yet.
    `x` and `y` must have known number of dimensions, and only the deterministic
    axes will be broadcasted.  You must ensure the non-deterministic axes are
    properly broadcasted by yourself.

    Args:
        x: The tensor `x`.
        y: The tensor `y`.
        axis: The axis to be concatenated.

    Returns:
        tf.Tensor: The broadcast and concatenated tensor.
    """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    # check the arguments
    x_static_shape = get_static_shape(x)
    if x_static_shape is None:
        raise ValueError('`x` with non-deterministic shape is not supported.')
    y_static_shape = get_static_shape(y)
    if y_static_shape is None:
        raise ValueError('`y` with non-deterministic shape is not supported.')

    x_rank = len(x_static_shape)
    y_rank = len(y_static_shape)
    out_ndims = max(x_rank, y_rank)
    min_axis = -out_ndims
    max_axis = out_ndims - 1

    if axis < min_axis or axis > max_axis:
        raise ValueError('Invalid axis: must >= {} and <= {}, got {}'.
                         format(min_axis, max_axis, axis))
    if axis >= 0:
        axis = axis - out_ndims

    # compute the broadcast shape
    out_static_shape = [None] * out_ndims

    x_tile = [1] * out_ndims
    y_tile = [1] * out_ndims
    assertions = []

    dynamic_shape_cache = {}

    def get_dynamic_shape(t):
        if t not in dynamic_shape_cache:
            dynamic_shape_cache[t] = get_shape(t)
        return dynamic_shape_cache[t]

    def broadcast_axis(i, a, b, a_tile, b_tile, a_tensor, b_tensor):
        err_msg = ('`x` and `y` cannot be broadcast concat: {} vs {}'.
                   format(x, y))

        # validate whether or not a == b or can be broadcasted
        if a is None and b is None:
            # both dynamic, must be equal
            a = get_dynamic_shape(a_tensor)[i]
            b = get_dynamic_shape(b_tensor)[i]
            assertions.append(tf.assert_equal(a, b, message=err_msg))

        elif a is not None and b is not None:
            # both static, check immediately
            if a != 1 and b != 1 and a != b:
                raise ValueError(err_msg)

            if a == 1:
                a_tile[i] = b
            elif b == 1:
                b_tile[i] = a

            out_static_shape[i] = max(a, b)

        elif a is None:
            # a dynamic, b can be 1 or equal to a
            a = get_dynamic_shape(a_tensor)[i]
            if b == 1:
                b_tile[i] = a
            else:
                assertions.append(tf.assert_equal(a, b, message=err_msg))
                out_static_shape[i] = b

        else:
            broadcast_axis(i, b, a, b_tile, a_tile, b_tensor, a_tensor)

    def maybe_prepend_dims(t, rank, name):
        if rank < out_ndims:
            t = prepend_dims(t, out_ndims - rank, name=name)
        return t

    def maybe_tile(t, tile, name):
        if any(s != 1 for s in tile):
            if any(is_tensor_object(s) for s in tile):
                tile = tf.stack(tile, axis=0)
            t = tf.tile(t, tile, name=name)
        return t

    with tf.name_scope(name, default_name='broadcast_concat', values=[x, y]):
        # infer the configurations
        for i in range(-1, -out_ndims - 1, -1):
            a = x_static_shape[i] if i >= -x_rank else 1
            b = y_static_shape[i] if i >= -y_rank else 1

            if i != axis:
                broadcast_axis(i, a, b, x_tile, y_tile, x, y)
            else:
                if a is not None and b is not None:
                    out_static_shape[i] = a + b

        # do broadcast
        x = maybe_tile(
            maybe_prepend_dims(x, x_rank, name='prepend_dims_to_x'),
            x_tile,
            name='tile_x'
        )
        y = maybe_tile(
            maybe_prepend_dims(y, y_rank, name='prepend_dims_to_y'),
            y_tile,
            name='tile_y'
        )

        with assert_deps(assertions) as asserted:
            if asserted:
                x = tf.identity(x)
                y = tf.identity(y)

        # do concat
        ret = tf.concat([x, y], axis=axis)
        ret.set_shape(tf.TensorShape(out_static_shape))
        return ret


@add_name_arg_doc
def transpose_conv2d_axis(input, from_channels_last, to_channels_last,
                          name=None):
    """
    Ensure the channels axis of `input` tensor to be placed at the desired axis.

    Args:
        input (tf.Tensor): The input tensor, at least 4-d.
        from_channels_last (bool): Whether or not the channels axis
            is the last axis in `input`? (i.e., the data format is "NHWC")
        to_channels_last (bool): Whether or not the channels axis
            should be the last axis in the output tensor?

    Returns:
        tf.Tensor: The (maybe) transposed output tensor.
    """
    if from_channels_last:
        input_spec = InputSpec(shape=('...', '?', '?', '?', '*'))
    else:
        input_spec = InputSpec(shape=('...', '?', '*', '?', '?'))
    input = input_spec.validate('input', input)
    input_shape = get_static_shape(input)
    sample_and_batch_axis = [i for i in range(len(input_shape) - 3)]

    # check whether or not axis should be transpose
    if from_channels_last and not to_channels_last:
        transpose_axis = [-1, -3, -2]
    elif not from_channels_last and to_channels_last:
        transpose_axis = [-2, -1, -3]
    else:
        transpose_axis = None

    # transpose the axis
    if transpose_axis is not None:
        transpose_axis = [i + len(input_shape) for i in transpose_axis]
        input = tf.transpose(input, sample_and_batch_axis + transpose_axis,
                             name=name or 'transpose_conv2d_axis')

    return input


@add_name_arg_doc
def transpose_conv2d_channels_last_to_x(input, channels_last, name=None):
    """
    Ensure the channels axis (known to be the last axis) of `input` tensor
    to be placed at the desired axis.

    Args:
        input (tf.Tensor): The input tensor, at least 4-d.
        channels_last (bool): Whether or not the channels axis
            should be the last axis in the output tensor?

    Returns:
        tf.Tensor: The (maybe) transposed output tensor.
    """
    return transpose_conv2d_axis(
        input, from_channels_last=True, to_channels_last=channels_last,
        name=name
    )


@add_name_arg_doc
def transpose_conv2d_channels_x_to_last(input, channels_last, name=None):
    """
    Ensure the channels axis of `input` tensor to be placed at the last axis.

    Args:
        input (tf.Tensor): The input tensor, at least 4-d.
        channels_last (bool): Whether or not the channels axis
            is the last axis in the `input` tensor?

    Returns:
        tf.Tensor: The (maybe) transposed output tensor.
    """
    return transpose_conv2d_axis(
        input, from_channels_last=channels_last, to_channels_last=True,
        name=name
    )


@add_name_arg_doc
def reshape_tail(input, ndims, shape, name=None):
    """
    Reshape the tail (last) `ndims` into specified `shape`.

    Usage::

        x = tf.zeros([2, 3, 4, 5, 6])
        reshape_tail(x, 3, [-1])  # output: zeros([2, 3, 120])
        reshape_tail(x, 1, [3, 2])  # output: zeros([2, 3, 4, 5, 3, 2])

    Args:
        input (Tensor): The input tensor, at least `ndims` dimensions.
        ndims (int): To reshape this number of dimensions at tail.
        shape (Iterable[int] or tf.Tensor): The shape of the new tail.

    Returns:
        tf.Tensor: The reshaped tensor.
    """
    input = tf.convert_to_tensor(input)
    if not is_tensor_object(shape):
        shape = list(int(s) for s in shape)
        neg_one_count = 0
        for s in shape:
            if s <= 0:
                if s == -1:
                    if neg_one_count > 0:
                        raise ValueError('`shape` is not a valid shape: at '
                                         'most one `-1` can be specified.')
                    else:
                        neg_one_count += 1
                else:
                    raise ValueError('`shape` is not a valid shape: {} is '
                                     'not allowed.'.format(s))

    with tf.name_scope(name or 'reshape_tail', values=[input]):
        # assert the dimension
        with assert_deps([
                    assert_rank_at_least(
                        input, ndims,
                        message='rank(input) must be at least ndims')
                ]) as asserted:
            if asserted:  # pragma: no cover
                input = tf.identity(input)

        # compute the static shape
        static_input_shape = get_static_shape(input)
        static_output_shape = None

        if static_input_shape is not None:
            if ndims > 0:
                left_shape = static_input_shape[:-ndims]
                right_shape = static_input_shape[-ndims:]
            else:
                left_shape = static_input_shape
                right_shape = ()

            # attempt to resolve "-1" in `shape`
            if isinstance(shape, list):
                if None not in right_shape:
                    shape_size = int(np.prod([s for s in shape if s != -1]))
                    right_shape_size = int(np.prod(right_shape))

                    if (-1 not in shape and shape_size != right_shape_size) or \
                            (-1 in shape and right_shape_size % shape_size != 0):
                        raise ValueError(
                            'Cannot reshape the tail dimensions of '
                            '`input` into `shape`: input {!r}, ndims '
                            '{}, shape {}.'.format(input, ndims, shape)
                        )

                    if -1 in shape:
                        pos = shape.index(-1)
                        shape[pos] = right_shape_size // shape_size

                static_output_shape = left_shape + \
                    tuple(s if s != -1 else None for s in shape)

        static_output_shape = tf.TensorShape(static_output_shape)

        # compute the dynamic shape
        input_shape = get_shape(input)
        if ndims > 0:
            output_shape = concat_shapes([input_shape[:-ndims], shape])
        else:
            output_shape = concat_shapes([input_shape, shape])

        # do reshape
        output = tf.reshape(input, output_shape)
        output.set_shape(static_output_shape)
        return output
