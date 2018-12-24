import tensorflow as tf

__all__ = ['default_kernel_initializer']


def default_kernel_initializer(weight_norm=False):
    """
    Get the default initializer for layer kernels (i.e., `W`s).

    Args:
        weight_norm (bool): Whether or not to apply weight normalization
            (Salimans & Kingma, 2016) on the kernel?  If :obj:`True`,
            will use ``tf.random_normal_initializer(0, .05)``.

    Returns:
        The default initializer for kernels.
    """
    if weight_norm:
        return tf.random_normal_initializer(0., .05)
    else:
        return tf.glorot_normal_initializer()
