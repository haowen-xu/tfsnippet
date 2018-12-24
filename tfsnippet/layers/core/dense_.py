import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (int_shape, flatten, unflatten, InputSpec,
                             ParamSpec, add_name_and_scope_arg_doc)
from ..initialization import default_kernel_initializer

__all__ = ['dense']


@add_arg_scope
@add_name_and_scope_arg_doc
def dense(input, units,
          activation_fn=None,
          normalizer_fn=None,
          weight_norm_fn=None,
          dtype=tf.float32,
          kernel=None,
          kernel_initializer=None,
          kernel_regularizer=None,
          kernel_constraint=None,
          use_bias=None,
          bias=None,
          bias_initializer=tf.zeros_initializer(),
          bias_regularizer=None,
          bias_constraint=None,
          trainable=True,
          name=None,
          scope=None):
    """
    Fully-connected layer.

    Roughly speaking, the dense layer is defined as::

        output = activation_fn(
            normalizer_fn(tf.matmul(input, weight_norm_fn(kernel)) + bias))

    Args:
        input: The input tensor, at least 2-d.
        units: Number of output units.
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        weight_norm_fn ((kernel, axis, scale_type) -> tf.Tensor):
            The weight normalization function.
            It can either be :func:`~tfsnippet.layers.weight_norm`,
            or some function that accepts ``(kernel, axis, scale_type)``
            arguments which then calls :func:`~tfsnippet.layers.weight_norm`.
        dtype: Data type of the parameters and input/output.
        kernel (Tensor): Instead of creating a new variable, use this tensor.
        kernel_initializer: The initializer for `kernel`.
            Would be ``default_kernel_initializer(...)`` if not specified.
        kernel_regularizer: The regularizer for `kernel`.
        kernel_constraint: The constraint for `kernel`.
        use_bias (bool or None): Whether or not to use `bias`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias (Tensor): Instead of creating a new variable, use this tensor.
        bias_initializer: The initializer for `bias`.
        bias_regularizer: The regularizer for `bias`.
        bias_constraint: The constraint for `bias`.
        trainable: Whether or not the parameters are trainable?

    Returns:
        tf.Tensor: The output tensor.
    """
    # check the arguments
    input_spec = InputSpec(shape=('...', '?', '*'), dtype=dtype)
    input = input_spec.validate(input)
    input_shape = int_shape(input)

    if use_bias is None:
        use_bias = normalizer_fn is None

    kernel_spec = ParamSpec(shape=(input_shape[-1], units), dtype=dtype)
    bias_spec = ParamSpec(shape=(units,), dtype=dtype)
    if kernel is not None:
        kernel = kernel_spec.validate(kernel)
    if bias is not None:
        bias = bias_spec.validate(bias)
    if kernel_initializer is None:
        kernel_initializer = default_kernel_initializer(
            use_weight_norm=weight_norm_fn is not None
        )

    # the main part of the dense layer
    with tf.variable_scope(scope, default_name=name or 'dense'):
        # create the variables
        if kernel is None:
            kernel = tf.get_variable(
                'kernel',
                shape=(input_shape[-1], units),
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
                constraint=kernel_constraint,
                trainable=trainable,
            )

        if weight_norm_fn is not None:
            kernel = weight_norm_fn(kernel, -1, None)

        if use_bias and bias is None:
            bias = tf.get_variable(
                'bias',
                shape=(units,),
                dtype=dtype,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint,
                trainable=trainable,
            )

        # do kernel * input + bias
        if len(int_shape(input)) == 2:
            output = tf.matmul(input, kernel)
            if use_bias:
                output = tf.nn.bias_add(output, bias)
        else:
            output, s1, s2 = flatten(input, 2)
            output = tf.matmul(input, kernel)
            if use_bias:
                output = tf.nn.bias_add(output, bias)
            output = unflatten(output, s1, s2)

        # apply the normalization function if specified
        if normalizer_fn is not None:
            output = normalizer_fn(output)

        # apply the activation function if specified
        if activation_fn is not None:
            output = activation_fn(output)

    return output
