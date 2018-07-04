import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.examples.utils import add_variable_scope

__all__ = ['dense']


@add_arg_scope
@add_variable_scope
def dense(inputs,
          units,
          activation_fn=None,
          normalizer_fn=None,
          kernel_initializer=None,
          bias_initializer=tf.zeros_initializer(),
          kernel_regularizer=None,
          bias_regularizer=None,
          name=None):
    output = tf.layers.dense(
        inputs=inputs,
        units=units,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_bias=normalizer_fn is None,
        name='linear'
    )
    if normalizer_fn is not None:
        output = normalizer_fn(output)
    if activation_fn is not None:
        output = activation_fn(output)
    return output
