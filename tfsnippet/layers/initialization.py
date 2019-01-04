import tensorflow as tf

__all__ = [
    'default_kernel_initializer',
    'get_variable_ddi',
]


def default_kernel_initializer(weight_norm=False):
    """
    Get the default initializer for layer kernels (i.e., `W`s).

    Args:
        weight_norm: Whether or not to apply weight normalization
            (Salimans & Kingma, 2016) on the kernel?  If is not :obj:`False`
            or :obj:`None`, will use ``tf.random_normal_initializer(0, .05)``.

    Returns:
        The default initializer for kernels.
    """
    if weight_norm not in (False, None):
        return tf.random_normal_initializer(0., .05)
    else:
        return tf.glorot_normal_initializer()


def get_variable_ddi(name,
                     initial_value,
                     shape=None,
                     dtype=tf.float32,
                     initializing=False,
                     regularizer=None,
                     constraint=None,
                     trainable=True,
                     **kwargs):
    """
    Wraps :func:`tf.get_variable` to support data-dependent initialization.

    Args:
        name: Name of the variable.
        initial_value: The data-dependent initial value of the variable.
        shape: Shape of the variable.
        dtype: Data type of the variable.
        initializing (bool): Whether or not it is building the graph for
            data-dependent initialization? Ignored if `initial_value` is absent.
        regularizer: Regularizer of the variable.
        constraint: Constraint of the variable.
        trainable (bool): Whether or not to the variable is trainable?
        \\**kwargs: Other named parameters passed to :func:`tf.get_variable`.

    Returns:
        tf.Variable or tf.Tensor: The variable or the tensor.
    """
    # TODO: detect shape from `initial_value` if not specified
    v = tf.get_variable(
        name, shape=shape, dtype=dtype, regularizer=regularizer,
        constraint=constraint, trainable=trainable,
        **kwargs
    )
    if initializing:
        v = v.assign(initial_value)
    return v


