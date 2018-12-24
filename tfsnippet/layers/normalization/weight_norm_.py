import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (ParamSpec, int_shape, get_default_scope_name,
                             add_name_and_scope_arg_doc)
from ..utils import validate_int_tuple_arg, resolve_negative_axis

__all__ = ['weight_norm']


@add_arg_scope
@add_name_and_scope_arg_doc
def weight_norm(kernel,
                axis,
                epsilon=1e-12,
                scale_type=None,
                log_scale=None,
                log_scale_initializer=None,
                log_scale_regularizer=None,
                log_scale_constraint=None,
                scale=None,
                scale_initializer=None,
                scale_regularizer=None,
                scale_constraint=None,
                trainable=True,
                name=None,
                scope=None):
    """
    Weight normalization proposed by (Salimans & Kingma, 2016).

    Roughly speaking, the weight normalization is defined as::

        if use_log_scale:
            scale = tf.exp(log_scale)
        kernel = scale * kernel / tf.reduce_mean(
            kernel, axis=<dimensions not in `axis`>, keepdims=True)

    This function does not support data-dependent initialization for `scale`
    or `log_scale`.  If you do need it, you have to keep `log_scale` and
    `scale` turned of, and use :func:`~tfsnippet.layers.act_norm` instead.

    Args:
        kernel: Tensor, the weight `w` to be normalized.
        axis (tuple[int]): The axis to apply weight normalization (See above).
        epsilon: Small float number to avoid dividing by zero.
        scale_type: One of {"log_scale", "scale", :obj:`None`}.
            If "log_scale", ``kernel = tf.exp(log_scale) * kernel / |kernel|``.
            If "scale", ``kernel = scale * kernel / |kernel|``.
            If :obj:`None`, kernel will not be scaled.
            Default is :obj:`None`.
        log_scale (Tensor): Instead of creating a new variable, use this tensor.
        log_scale_initializer: The initializer for `log_scale`.
        log_scale_regularizer: The regularizer for `log_scale`.
        log_scale_constraint: The constraint for `log_scale`.
        scale (Tensor): Instead of creating a new variable, use this tensor.
        scale_initializer: The initializer for `scale`.
        scale_regularizer: The regularizer for `scale`.
        scale_constraint: The constraint for `scale`.
        trainable (bool): Whether or not `log_scale` and `scale` are trainable?
    """
    # check the parameters
    if scale_type not in ('log_scale', 'scale', None):
        raise ValueError('`scale_type` must be one of {{"log_scale", "scale", '
                         'None}}: got {!r}.'.format(scale_type))
    if scale_type == 'log_scale' and scale is not None:
        raise ValueError('`scale_type` is "log_scale", but `scale` is '
                         'specified.')
    if scale_type == 'scale' and log_scale is not None:
        raise ValueError('`scale_type` is "scale", but `log_scale` is '
                         'specified.')

    kernel = tf.convert_to_tensor(kernel)
    kernel_shape = int_shape(kernel)
    dtype = kernel.dtype.base_dtype
    var_spec = ParamSpec(kernel_shape, dtype=dtype)

    if log_scale_initializer is None:
        log_scale_initializer = tf.zeros_initializer(dtype=dtype)
    if scale_initializer is None:
        scale_initializer = tf.ones_initializer(dtype=dtype)
    if log_scale is not None:
        log_scale = var_spec.validate(log_scale)
    if scale is not None:
        scale = var_spec.validate(scale)

    # any dimension not specified in `axis` should be averaged out
    axis = resolve_negative_axis(
        len(kernel_shape), validate_int_tuple_arg('axis', axis))
    reduce_axis = tuple(a for a in range(len(kernel_shape)) if a not in axis)

    with tf.variable_scope(
            scope, default_name=get_default_scope_name(name or 'weight_norm')):
        # normalize the kernel
        kernel = tf.nn.l2_normalize(kernel, axis=reduce_axis, epsilon=epsilon)

        # create the scaling variable
        if scale_type == 'log_scale':
            if log_scale is None:
                log_scale = tf.get_variable(
                    'log_scale',
                    shape=kernel_shape,
                    dtype=dtype,
                    initializer=log_scale_initializer,
                    regularizer=log_scale_regularizer,
                    constraint=log_scale_constraint,
                    trainable=trainable
                )
            kernel = kernel * tf.exp(log_scale)

        elif scale_type == 'scale':
            if scale is None:
                scale = tf.get_variable(
                    'scale',
                    shape=kernel_shape,
                    dtype=dtype,
                    initializer=scale_initializer,
                    regularizer=scale_regularizer,
                    constraint=scale_constraint,
                    trainable=trainable
                )
            kernel = kernel * scale

        # now return the normalized weight
        return kernel
