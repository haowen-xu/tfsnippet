import functools

import tensorflow as tf

from .doc_utils import add_name_arg_doc
from .type_utils import is_tensor_object

__all__ = [
    'get_static_shape', 'get_batch_size', 'get_rank', 'get_shape',
    'get_dimension_size', 'get_dimensions_size', 'resolve_negative_axis',
    'concat_shapes', 'is_shape_equal',
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
    if len(set(ret)) != len(ret):
        raise ValueError('`axis` has duplicated elements after resolving '
                         'negative axis: ndims {}, axis {}.'.
                         format(ndims, axis))
    return tuple(ret)


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
    tensor_shape = get_static_shape(tensor)
    if tensor_shape is not None:
        return len(tensor_shape)
    return tf.rank(tensor, name=name)


@add_name_arg_doc
def get_dimension_size(tensor, axis, name=None):
    """
    Get the size of `tensor` of specified `axis`.

    Args:
        tensor (tf.Tensor): The tensor to be tested.
        axis (Iterable[int] or None): The dimension to be queried.

    Returns:
        int or tf.Tensor: An integer or a tensor, the size of queried dimension.
    """
    tensor = tf.convert_to_tensor(tensor)

    with tf.name_scope(name, default_name='get_dimension_size',
                       values=[tensor]):
        shape = get_static_shape(tensor)

        if shape is not None and not is_tensor_object(axis) and \
                shape[axis] is not None:
            return shape[axis]

        return tf.shape(tensor)[axis]


@add_name_arg_doc
def get_dimensions_size(tensor, axes=None, name=None):
    """
    Get the size of `tensor` of specified `axes`.

    If `axes` is :obj:`None`, select the size of all dimensions.

    Args:
        tensor (tf.Tensor): The tensor to be tested.
        axes (Iterable[int] or None): The dimensions to be selected.

    Returns:
        tuple[int] or tf.Tensor: A tuple of integers if all selected
            dimensions have static sizes.  Otherwise a tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    if axes is not None:
        axes = tuple(axes)
        if not axes:
            return ()

    with tf.name_scope(name, default_name='get_dimensions_size',
                       values=[tensor]):
        shape = get_static_shape(tensor)

        if shape is not None and axes is not None:
            shape = tuple(shape[a] for a in axes)

        if shape is None or None in shape:
            dynamic_shape = tf.shape(tensor)
            if axes is None:
                shape = dynamic_shape
            else:
                shape = tf.stack([dynamic_shape[i] for i in axes], axis=0)

        return shape


get_shape = functools.partial(get_dimensions_size, axes=None)


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


@add_name_arg_doc
def is_shape_equal(x, y, name=None):
    """
    Check whether the shape of `x` equals to `y`.

    Args:
        x: A tensor.
        y: Another tensor, to compare with `x`.

    Returns:
        bool or tf.Tensor: The static or dynamic comparison result.
    """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    x_shape = get_static_shape(x)
    y_shape = get_static_shape(y)

    # both shapes have deterministic dimensions, we can perform a fast check
    if x_shape is not None and y_shape is not None:
        # dimension mismatch, cannot be equal
        if len(x_shape) != len(y_shape):
            return False

        # gather the axis to check
        axis_to_check = []
        for i, (a, b) in enumerate(zip(x_shape, y_shape)):
            if a is None or b is None:
                axis_to_check.append(i)
            else:
                if a != b:
                    return False

        # no dynamic axis to check, confirm equality
        if not axis_to_check:
            return True

        # generate the dynamic check
        with tf.name_scope(name or 'is_shape_equal', values=[x, y]):
            x_shape = get_shape(x)
            y_shape = get_shape(y)
            return tf.reduce_all([tf.equal(x_shape[a], y_shape[a])
                                  for a in axis_to_check])

    # either one of the shapes has non-deterministic dimensions
    with tf.name_scope(name or 'is_shape_equal', values=[x, y]):
        x_shape = get_shape(x)
        y_shape = get_shape(y)
        return tf.cond(
            tf.equal(tf.rank(x), tf.rank(y)),
            lambda: tf.reduce_all(
                tf.equal(
                    tf.concat([x_shape, y_shape], axis=0),
                    tf.concat([y_shape, x_shape], axis=0)
                )
            ),
            lambda: tf.constant(False, dtype=tf.bool)
        )
