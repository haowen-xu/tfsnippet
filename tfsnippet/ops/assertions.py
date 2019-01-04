import tensorflow as tf

from tfsnippet.utils import add_name_arg_doc, is_tensor_object, get_static_shape

__all__ = [
    'assert_scalar_equal',
    'assert_rank',
    'assert_rank_at_least',
]


def _make_assertion_error(expected, actual, message=None):
    ret = 'Assertion failed for {}: {}'.format(expected, actual)
    if message:
        ret += '; {}'.format(message)
    return AssertionError(ret)


@add_name_arg_doc
def assert_scalar_equal(a, b, message=None, name=None):
    """
    Assert 0-d scalar `a` == `b`.

    Args:
        a: A 0-d tensor.
        b: A 0-d tensor.
        message: Message to display when assertion failed.

    Returns:
        tf.Operation or None: The TensorFlow assertion operation,
            or None if can be statically asserted.
    """
    if not is_tensor_object(a) and not is_tensor_object(b):
        if a != b:
            raise _make_assertion_error(
                'a == b', '{!r} != {!r}'.format(a, b), message)
    else:
        return tf.assert_equal(a, b, message=message, name=name)


@add_name_arg_doc
def assert_rank(x, ndims, message=None, name=None):
    """
    Assert the rank of `x` is `ndims`.

    Args:
        x: A tensor.
        ndims (int or tf.Tensor): An integer, or a 0-d integer tensor.
        message: Message to display when assertion failed.

    Returns:
        tf.Operation or None: The TensorFlow assertion operation,
            or None if can be statically asserted.
    """
    if not is_tensor_object(ndims) and get_static_shape(x) is not None:
        ndims = int(ndims)
        x_ndims = len(get_static_shape(x))
        if x_ndims != ndims:
            raise _make_assertion_error(
                'rank(x) == ndims', '{!r} != {!r}'.format(x_ndims, ndims),
                message
            )
    else:
        return tf.assert_rank(x, ndims, message=message, name=name)


@add_name_arg_doc
def assert_rank_at_least(x, ndims, message=None, name=None):
    """
    Assert the rank of `x` is at least `ndims`.

    Args:
        x: A tensor.
        ndims (int or tf.Tensor): An integer, or a 0-d integer tensor.
        message: Message to display when assertion failed.

    Returns:
        tf.Operation or None: The TensorFlow assertion operation,
            or None if can be statically asserted.
    """
    x = tf.convert_to_tensor(x)
    if not is_tensor_object(ndims) and get_static_shape(x) is not None:
        ndims = int(ndims)
        x_ndims = len(get_static_shape(x))
        if x_ndims < ndims:
            raise _make_assertion_error(
                'rank(x) >= ndims', '{!r} < {!r}'.format(x_ndims, ndims),
                message
            )
    else:
        return tf.assert_rank_at_least(x, ndims, message=message, name=name)
