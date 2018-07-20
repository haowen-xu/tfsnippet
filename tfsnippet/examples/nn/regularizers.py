import tensorflow as tf

__all__ = ['regularization_loss', 'l2_regularizer']


def regularization_loss():
    """
    Sum up all the regularization losses in the graph collection
    ``REGULARIZATION_LOSSES``.

    Returns:
        tf.Tensor: The total regularization loss.
    """
    losses = tf.get_collection_ref(tf.GraphKeys.REGULARIZATION_LOSSES)
    if not losses:
        return 0
    elif len(losses) == 1:
        return losses[0]
    else:
        return sum(losses[1:], losses[0])


def l2_regularizer(lambda_):
    """
    Get an L2 regularizer.

    Args:
        lambda_: The coefficiency of L2 regularizer.

    Returns:
        (tf.Tensor) -> tf.Tensor: A function computes the L2 regularization
            term for input tensor.
    """
    def regularizer(x):
        return tf.reduce_sum(tf.square(x)) * lambda_
    return regularizer
