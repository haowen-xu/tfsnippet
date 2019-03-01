import copy

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import maybe_add_histogram

__all__ = ['as_gated']


def as_gated(layer_fn, sigmoid_bias=2., default_name=None):
    """
    Wrap a layer function into a gated layer function.

    For example, the following `gated_dense` function::

        @add_arg_scope
        def gated_dense(inputs, units, activation_fn=None, sigmoid_bias=2.,
                        name=None, scope=None, **kwargs):
            with tf.name_scope(scope, default_name=name):
                gate = tf.sigmoid(sigmoid_bias +
                    dense(inputs, units, scope='gate', **kwargs))
                return gate * dense(
                    inputs, units, activation_fn=activation_fn, scope='main',
                    **kwargs
                )

    can be deduced by applying this function::

        gated_dense = as_gated(dense)

    Args:
        layer_fn: The layer function to be wrapped.
        sigmoid_bias: The constant bias added to the `gate` before
            applying the sigmoid activation.
        default_name: Default name of variable scope.

    Returns:
        The wrapped gated layer function.

    Notes:
        If a layer supports `gated` argument (e.g., :func:`spt.layers.dense`),
        it is generally better to use that argument, instead of using this
        :func:`as_gated` wrapper on the layer.
    """
    if not default_name:
        if getattr(layer_fn, '__name__', None):
            default_name = 'gated_' + layer_fn.__name__
    if not default_name:
        raise ValueError('`default_name` cannot be inferred, you must specify '
                         'this argument.')

    @add_arg_scope
    def gated_layer(*args, **kwargs):
        name = kwargs.pop('name', None)
        scope = kwargs.pop('scope', None)
        activation_fn = kwargs.pop('activation_fn', None)

        with tf.variable_scope(scope, default_name=name or default_name):
            # the gate branch
            gate_kwargs = copy.copy(kwargs)
            gate_kwargs['scope'] = 'gate'
            gate = tf.sigmoid(sigmoid_bias + layer_fn(*args, **gate_kwargs))

            # the main branch
            main_kwargs = copy.copy(kwargs)
            main_kwargs['scope'] = 'main'
            main_kwargs['activation_fn'] = activation_fn
            main = layer_fn(*args, **main_kwargs)

            # compose the final output
            output = main * gate
            maybe_add_histogram(output, 'output')

        return output

    return gated_layer
