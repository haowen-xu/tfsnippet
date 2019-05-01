import numpy as np
import tensorflow as tf

from .type_utils import convert_to_tensor_and_cast

__all__ = ['bits_per_dimension']


def bits_per_dimension(log_p, value_size, scale=256., name=None):
    """
    Compute "bits per dimension" of `x`.

    `BPD(x) = - log(p(x)) / (log(2) * Dim(x))`

    If `u = s * x`, then:

    `BPD(x) = - (log(p(u)) - log(s) * Dim(x)) / (log(2) * Dim(x))`

    Args:
        log_p (Tensor): If `scale` is specified, then it should be `log(p(u))`.
            Otherwise it should be `log(p(x))`.
        value_size (int or Tensor): The size of each `x`, i.e., `Dim(x)`.
        scale (float or Tensor or None): The scale `s`, where `u = s * x`,
            and `log_p` is `log(p(u))`.

    Returns:
        tf.Tensor: The computed "bits per dimension" of `x`.
    """
    log_p = tf.convert_to_tensor(log_p)
    dtype = log_p.dtype.base_dtype

    with tf.name_scope(name, default_name='bits_per_dimension', values=[log_p]):
        if scale is not None:
            scale = convert_to_tensor_and_cast(scale, dtype)
            nll = tf.log(scale) * value_size - log_p
        else:
            nll = -log_p
        ret = nll / (np.log(2) * value_size)

    return ret
