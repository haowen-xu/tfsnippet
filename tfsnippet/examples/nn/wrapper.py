import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.examples.utils import (add_variable_scope,
                                      validate_strides_or_kernel_size)

__all__ = [
    'dense', 'conv2d', 'deconv2d', 'batch_norm_2d',
]


@add_arg_scope
@add_variable_scope
def dense(inputs,
          units,
          activation_fn=None,
          normalizer_fn=None,
          use_bias=True,
          kernel_initializer=None,
          bias_initializer=tf.zeros_initializer(),
          kernel_regularizer=None,
          bias_regularizer=None,
          kernel_constraint=None,
          bias_constraint=None,
          trainable=True,
          name=None):
    output = tf.layers.dense(
        inputs=inputs,
        units=units,
        use_bias=use_bias and (normalizer_fn is None),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name='activation'
    )
    if normalizer_fn is not None:
        output = normalizer_fn(output)
    if activation_fn is not None:
        output = activation_fn(output)
    return output


@add_arg_scope
@add_variable_scope
def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           padding='valid',
           channels_last=False,
           use_bias=True,
           activation_fn=None,
           normalizer_fn=None,
           kernel_initializer=None,
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None):
    output = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=validate_strides_or_kernel_size('kernel_size', kernel_size),
        strides=validate_strides_or_kernel_size('strides', strides),
        padding=padding,
        data_format='channels_last' if channels_last else 'channels_first',
        use_bias=use_bias and (normalizer_fn is None),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name='activation'
    )
    if normalizer_fn is not None:
        output = normalizer_fn(output)
    if activation_fn is not None:
        output = activation_fn(output)
    return output


@add_arg_scope
@add_variable_scope
def deconv2d(inputs,
             filters,
             kernel_size,
             strides=(1, 1),
             padding='valid',
             channels_last=False,
             use_bias=True,
             activation_fn=None,
             normalizer_fn=None,
             kernel_initializer=None,
             bias_initializer=tf.zeros_initializer(),
             kernel_regularizer=None,
             bias_regularizer=None,
             kernel_constraint=None,
             bias_constraint=None,
             trainable=True,
             name=None):
    output = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=validate_strides_or_kernel_size('kernel_size', kernel_size),
        strides=validate_strides_or_kernel_size('strides', strides),
        padding=padding,
        data_format='channels_last' if channels_last else 'channels_first',
        use_bias=use_bias and (normalizer_fn is None),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name='activation'
    )
    if normalizer_fn is not None:
        output = normalizer_fn(output)
    if activation_fn is not None:
        output = activation_fn(output)
    return output


@add_arg_scope
def batch_norm_2d(inputs,
                  channels_last=False,
                  momentum=0.99,
                  epsilon=1e-3,
                  center=True,
                  scale=True,
                  beta_initializer=tf.zeros_initializer(),
                  gamma_initializer=tf.ones_initializer(),
                  moving_mean_initializer=tf.zeros_initializer(),
                  moving_variance_initializer=tf.ones_initializer(),
                  beta_regularizer=None,
                  gamma_regularizer=None,
                  beta_constraint=None,
                  gamma_constraint=None,
                  training=False,
                  trainable=True,
                  name=None):
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1 if channels_last else 1,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        trainable=trainable,
        training=training,
        name=name
    )
