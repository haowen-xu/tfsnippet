import tensorflow as tf

__all__ = ['convert_to_tensor_and_cast']


def convert_to_tensor_and_cast(x, dtype=None):
    """
    Convert `x` into a :class:`tf.Tensor`, and cast its dtype if required.

    Args:
        x: The tensor to be converted into a :class:`tf.Tensor`.
        dtype (tf.DType): The data type.

    Returns:
        tf.Tensor: The converted and casted tensor.
    """
    x = tf.convert_to_tensor(x)
    if dtype is not None:
        dtype = tf.as_dtype(dtype)
        if dtype != x.dtype:
            x = tf.cast(x, dtype)
    return x
