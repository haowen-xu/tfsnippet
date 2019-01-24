import tensorflow as tf

from tfsnippet.utils import add_name_arg_doc

__all__ = ['l2_regularizer']


@add_name_arg_doc
def l2_regularizer(lambda_, name=None):
    """
    Construct an L2 regularizer that computes the L2 regularization loss::

        output = lambda_ * 0.5 * sum(input ** 2)

    Args:
        lambda_ (float or Tensor or None): The coefficiency of L2 regularizer.
            If `lambda_` is :obj:`None`, will return :obj:`None`.

    Returns:
        (tf.Tensor) -> tf.Tensor: A function that computes the L2
            regularization term for input tensor.  Will be :obj:`None`
            if `lambda_` is :obj:`None`.
    """
    if lambda_ is None:
        return None

    def regularizer(input):
        input = tf.convert_to_tensor(input)
        with tf.name_scope(name, default_name='l2_regularization',
                           values=[input]):
            return lambda_ * tf.nn.l2_loss(input)

    return regularizer
