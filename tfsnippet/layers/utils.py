from tfsnippet.utils import is_integer

__all__ = ['validate_int_tuple_arg', 'resolve_negative_axis']


def validate_int_tuple_arg(arg_name, value):
    """
    Validate the specified argument as a tuple of integers.

    Args:
        arg_name (str): Name of the argument.
        value (int or Iterable[int]): An integer, or an iterable collection
            of integers, to be casted into tuples of integers.

    Returns:
        tuple[int]: The tuple of integers.
    """
    if is_integer(value):
        value = (value,)
    else:
        try:
            value = tuple(int(v) for v in value)
        except (ValueError, TypeError):
            raise TypeError('`{}` cannot be converted to a tuple of integers: '
                            '{!r}'.format(arg_name, value))
    return value


def resolve_negative_axis(ndims, axis):
    """
    Resolve all negative `axis` indices.

    Args:
        ndims (int): Number of total dimensions.
        axis (tuple[int]): The axis indices.

    Returns:
        tuple[int]: The resolved axis indices.
    """
    axis = tuple(int(a) for a in axis)
    ret = []
    for a in axis:
        a += ndims
        if a < 0 or a >= ndims:
            raise ValueError('`axis` out of range: {} vs ndims {}.'.
                             format(axis, ndims))
        ret.append(a)
    return tuple(ret)
