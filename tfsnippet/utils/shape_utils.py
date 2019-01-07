import functools

import tensorflow as tf

from .debugging import assert_deps
from .doc_utils import add_name_arg_doc
from .type_utils import is_tensor_object

__all__ = [
    'get_static_shape', 'resolve_negative_axis',
    'flatten', 'unflatten',
    'get_batch_size', 'get_rank', 'get_shape', 'get_dimensions_size',
    'concat_shapes',
    'broadcast_to_shape',
    'transpose_conv2d_axis', 'transpose_conv2d_channels_last_to_x',
    'transpose_conv2d_channels_x_to_last', 'reshape_conv2d_to_dense',
]


def get_static_shape(tensor):
    """
    Get the the static shape of specified `tensor` as a tuple.

    Args:
        tensor: The tensor object.

    Returns:
        tuple[int or None] or None: The static shape tuple, or :obj:`None`
            if the dimensions of `tensor` is not deterministic.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tensor.get_shape()
    if shape.ndims is None:
        shape = None
    else:
        shape = tuple((int(v) if v is not None else None)
                      for v in shape.as_list())
    return shape


def resolve_negative_axis(ndims, axis):
    """
    Resolve all negative `axis` indices according to `ndims` into positive.

    Usage::

        resolve_negative_axis(4, [0, -1, -2])  # output: (0, 3, 2)

    Args:
        ndims (int): Number of total dimensions.
        axis (Iterable[int]): The axis indices.

    Returns:
        tuple[int]: The resolved positive axis indices.

    Raises:
        ValueError: If any index in `axis` is out of range.
    """
    axis = tuple(int(a) for a in axis)
    ret = []
    for a in axis:
        if a < 0:
            a += ndims
        if a < 0 or a >= ndims:
            raise ValueError('`axis` out of range: {} vs ndims {}.'.
                             format(axis, ndims))
        ret.append(a)
    return tuple(ret)


@add_name_arg_doc
def flatten(x, k, name=None):
    """
    Flatten the front dimensions of `x`, such that the resulting tensor
    will have at most `k` dimensions.

    Args:
        x (Tensor): The tensor to be flatten.
        k (int): The maximum number of dimensions for the resulting tensor.

    Returns:
        (tf.Tensor, tuple[int or None], tuple[int] or tf.Tensor) or (tf.Tensor, None, None):
            (The flatten tensor, the static front shape, and the front shape),
            or (the original tensor, None, None)
    """
    x = tf.convert_to_tensor(x)
    if k < 1:
        raise ValueError('`k` must be greater or equal to 1.')
    if not x.get_shape():
        raise ValueError('`x` is required to have known number of '
                         'dimensions.')
    shape = get_static_shape(x)
    if len(shape) < k:
        raise ValueError('`k` is {}, but `x` only has rank {}.'.
                         format(k, len(shape)))
    if len(shape) == k:
        return x, None, None

    with tf.name_scope(name, default_name='flatten', values=[x]):
        if k == 1:
            static_shape = shape
            if None in shape:
                shape = tf.shape(x)
            return tf.reshape(x, [-1]), static_shape, shape
        else:
            front_shape, back_shape = shape[:-(k-1)], shape[-(k-1):]
            static_front_shape = front_shape
            static_back_shape = back_shape
            if None in front_shape or None in back_shape:
                dynamic_shape = tf.shape(x)
                if None in front_shape:
                    front_shape = dynamic_shape[:-(k-1)]
                if None in back_shape:
                    back_shape = dynamic_shape[-(k-1):]
            if isinstance(back_shape, tuple):
                x = tf.reshape(x, [-1] + list(back_shape))
            else:
                x = tf.reshape(x, tf.concat([[-1], back_shape], axis=0))
                x.set_shape(tf.TensorShape([None] + list(static_back_shape)))
            return x, static_front_shape, front_shape


@add_name_arg_doc
def unflatten(x, static_front_shape, front_shape, name=None):
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


@add_name_arg_doc
def get_batch_size(tensor, axis=0, name=None):
    """
    Infer the mini-batch size according to `tensor`.

    Args:
        tensor (tf.Tensor): The input placeholder.
        axis (int): The axis of mini-batches.  Default is 0.

    Returns:
        int or tf.Tensor: The batch size.
    """
    tensor = tf.convert_to_tensor(tensor)
    axis = int(axis)
    with tf.name_scope(name, default_name='get_batch_size', values=[tensor]):
        batch_size = None
        shape = get_static_shape(tensor)
        if shape is not None:
            batch_size = shape[axis]
        if batch_size is None:
            batch_size = tf.shape(tensor)[axis]
    return batch_size


@add_name_arg_doc
def get_rank(tensor, name=None):
    """
    Get the rank of the tensor.

    Args:
        tensor (tf.Tensor): The tensor to be tested.
        name: TensorFlow name scope of the graph nodes.

    Returns:
        int or tf.Tensor: The rank.
    """
    tensor = tf.convert_to_tensor(tensor)
    tensor_shape = get_static_shape(tensor)
    if tensor_shape is not None:
        return len(tensor_shape)
    return tf.rank(tensor, name=name)


@add_name_arg_doc
def get_dimensions_size(tensor, axis=None, name=None):
    """
    Get the size of `tensor` of specified `axis`.

    If `axis` is :obj:`None`, select the size of all dimensions.

    Args:
        tensor (tf.Tensor): The tensor to be tested.
        axis (Iterable[int] or None): The dimensions to be selected.

    Returns:
        tuple[int] or tf.Tensor: A tuple of integers if all selected
            dimensions have static sizes.  Otherwise a tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    if axis is not None:
        axis = tuple(axis)
        if not axis:
            return ()

    with tf.name_scope(name, default_name='get_dimensions_size',
                       values=[tensor]):
        shape = get_static_shape(tensor)

        if shape is not None and axis is not None:
            shape = tuple(shape[a] for a in axis)

        if shape is None or None in shape:
            dynamic_shape = tf.shape(tensor)
            if axis is None:
                shape = dynamic_shape
            else:
                shape = tf.stack([dynamic_shape[i] for i in axis], axis=0)

        return shape


get_shape = functools.partial(get_dimensions_size, axis=None)


@add_name_arg_doc
def concat_shapes(shapes, name=None):
    """
    Concat shapes from `shapes`.

    Args:
        shapes (Iterable[tuple[int] or tf.Tensor]): List of shape tuples
            or tensors.

    Returns:
        tuple[int] or tf.Tensor: The concatenated shape.
    """
    shapes = tuple(shapes)
    if any(is_tensor_object(s) for s in shapes):
        shapes = [
            s if is_tensor_object(s) else tf.constant(s, dtype=tf.int32)
            for s in shapes
        ]
        with tf.name_scope(name, default_name='concat_shapes', values=shapes):
            return tf.concat(shapes, axis=0)
    else:
        return sum((tuple(s) for s in shapes), ())


def broadcast_to_shape_sub(x, shape, cannot_broadcast_msg):
    """
    A sub-procedure to broadcast the shape of `x` to match `shape`.
    If the length of `shape` is lower than the dimension of `x`, only
    the tail of `x` will be matched.

    Args:
        x: A tensor.
        shape (tuple[int] or tf.Tensor): Broadcast `x` to match this shape.
        cannot_broadcast_msg (str): Error message to display when `x`
            cannot broadcast to match `shape`.

    Returns:
        tf.Tensor: The broadcasted tensor.
    """
    from tfsnippet.ops import smart_cond
    x_shape = get_static_shape(x)

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
        # we have no ideal about what `shape` is here, thus we need to assert
        # the shape after ``x * ones(shape)``.
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
def broadcast_to_shape(x, shape, name=None):
    """
    Broadcast `x` to match `shape`.

    Args:
        x: A tensor.
        shape (tuple[int] or tf.Tensor): Broadcast `x` to match this shape.
            It must hold that ``rank(x) <= len(shape)``.

    Returns:
        tf.Tensor: The broadcasted tensor.
    """
    from tfsnippet.ops import assert_rank

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
            with assert_deps([tf.assert_less_equal(
                    tf.rank(x), len(shape), message=cannot_broadcast_msg)]):
                x = tf.identity(x)
        else:
            with assert_deps([assert_rank(
                    shape, 1, message=cannot_broadcast_msg)]):
                shape = tf.identity(shape)
            with assert_deps([tf.assert_less_equal(
                    tf.rank(x), tf.size(shape), message=cannot_broadcast_msg)]):
                x = tf.identity(x)

        # do broadcast
        return broadcast_to_shape_sub(x, shape, cannot_broadcast_msg)


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
    from .tensor_spec import InputSpec
    if from_channels_last:
        input_spec = InputSpec(shape=('...', '?', '?', '?', '*'))
    else:
        input_spec = InputSpec(shape=('...', '?', '*', '?', '?'))
    input = input_spec.validate(input)
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
def reshape_conv2d_to_dense(input, name=None):
    """
    Flatten the last three axis of `input` into one dimension.

    This operation is generally used to reshape 2-d convolution outputs
    to dense layer inputs, which is the origin of the method name.

    Args:
        input: The input tensor.

    Returns:
        tf.Tensor: The output tensor.
    """
    from .tensor_spec import InputSpec

    input_spec = InputSpec(shape=('...', '?', '?', '?', '?'))
    input = input_spec.validate(input)

    with tf.name_scope(name or 'reshape_conv2d_to_dense', values=[input]):
        input_shape = get_static_shape(input)

        # inspect the static shape
        left_shape = input_shape[:-3]
        right_shape = input_shape[-3:]

        if any(i is None for i in right_shape):
            static_shape = left_shape + (None,)
        else:
            static_shape = left_shape + (int(np.prod(right_shape)),)
        static_shape = tf.TensorShape(static_shape)

        # inspect the dynamic shape
        if any(i is None for i in left_shape):
            left_shape = get_shape(input)[:-3]
            shape = tf.concat([left_shape, [-1]], axis=0)
        else:
            shape = left_shape + (-1,)

        # now reshape the tensor
        output = tf.reshape(input, shape)
        output.set_shape(static_shape)

    return output
