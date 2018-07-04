__all__ = ['int_shape']


def int_shape(tensor):
    """
    Get the int shape tuple of specified `tensor`.

    Args:
        tensor: The tensor object.

    Returns:
        tuple[int or None]: The int shape tuple.
    """
    shape = tensor.get_shape().as_list()
    return tuple((int(v) if v is not None else None) for v in shape)
