import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.ops import smart_cond, convert_to_tensor_and_cast
from tfsnippet.utils import add_name_arg_doc, get_shape

__all__ = ['dropout']


@add_arg_scope
@add_name_arg_doc
def dropout(input, rate=.5, noise_shape=None, training=False, name=None):
    """
    Apply dropout on `input`.

    Args:
        input (Tensor): The input tensor.
        rate (float or tf.Tensor): The rate of dropout.
        noise_shape (tuple[int] or tf.Tensor): Shape of the noise.
            If not specified, use the shape of `input`.
        training (bool or tf.Tensor): Whether or not the model is under
            training stage?

    Returns:
        tf.Tensor: The dropout transformed tensor.
    """
    input = tf.convert_to_tensor(input)

    with tf.name_scope(name, default_name='dropout', values=[input]):
        dtype = input.dtype.base_dtype
        retain_prob = convert_to_tensor_and_cast(1. - rate, dtype=dtype)
        inv_retain_prob = 1. / retain_prob
        if noise_shape is None:
            noise_shape = get_shape(input)

        def training_branch():
            noise = tf.random_uniform(
                shape=noise_shape, minval=0., maxval=1., dtype=dtype)
            mask = tf.cast(noise < retain_prob, dtype=dtype)
            return input * mask * inv_retain_prob

        def testing_branch():
            return input

        return smart_cond(
            training,
            training_branch,
            testing_branch,
        )
