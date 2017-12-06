import functools
import tensorflow as tf
from tensorflow.contrib import layers

from .lambda_ import Lambda

__all__ = ['Dense', 'Linear']


class Dense(Lambda):
    """
    Dense (fully-connected) layer :class:`~tfsnippet.modules.Module`.

    This class wraps :func:`~tf.contrib.layers.fully_connected` function into
    a :class:`~tfsnippet.modules.Module`, which reuses the weights and biases.

    Args:
        num_outputs (int): The number of output units in the layer.
        activation_fn (tf.Tensor -> tf.Tensor or None):
            Activation function (default :func:`~tf.nn.relu`).
            Explicitly set it to :obj:`None` to skip it and obtain a linear
            activation.
        normalizer_fn:
            Normalization function to apply. If `normalizer_fn` is provided,
            `biases_initializer` and `biases_regularizer` are ignored and
            `biases` are not created nor added. (default :obj:`None`)
        normalizer_params (dict[str, any]):
            Normalization function named arguments. (default :obj:`None`)
        weights_initializer:
            Initializer for the weights.
            (default :func:`~tf.contrib.layers.xavier_initializer`)
        weights_regularizer:
            Optional regularizer for the weights. (default :obj:`None`)
        biases_initializer:
            Initializer for the biases. If :obj:`None` skip biases.
            (default :func:`~tf.zero_initializer`)
        biases_regularizer:
            Optional regularizer for the biases. (default :obj:`None`)
        trainable (bool):
            Whether or not to add the variables to the graph collection
            ``tf.GraphKeys.TRAINABLE_VARIABLES``? (default :obj:`True`)
        name (str):
            Optional name of this module
            (argument of :class:`~tfsippet.scaffold.VarScopeObject`).
        scope (str):
            Optional scope of this module
            (argument of :class:`~tfsippet.scaffold.VarScopeObject`).
    """

    def __init__(self,
                 num_outputs,
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,
                 normalizer_params=None,
                 weights_initializer=layers.xavier_initializer(),
                 weights_regularizer=None,
                 biases_initializer=tf.zeros_initializer(),
                 biases_regularizer=None,
                 trainable=True,
                 scope=None,
                 name=None):
        super(Dense, self).__init__(
            functools.partial(
                layers.fully_connected,
                num_outputs=num_outputs,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=biases_initializer,
                biases_regularizer=biases_regularizer,
                trainable=trainable,
            ),
            name=name,
            scope=scope,
        )


class Linear(Dense):
    """
    Linear layer :class:`~tfsnippet.modules.Module`.

    Args:
        num_outputs (int): The number of output units in the layer.
        normalizer_fn: Normalization function to apply.
                       If `normalizer_fn` is provided, `biases_initializer` and
                       `biases_regularizer` are ignored and `biases` are not
                       created nor added. (default :obj:`None`)
        normalizer_params (dict[str, any]): Normalization function named
                                            arguments. (default :obj:`None`)
        weights_initializer: Initializer for the weights. (default
                             :func:`~tf.contrib.layers.xavier_initializer`)
        weights_regularizer: Optional regularizer for the weights.
                             (default :obj:`None`)
        biases_initializer: Initializer for the biases. If :obj:`None` skip
                            biases. (default :func:`~tf.zero_initializer`)
        biases_regularizer: Optional regularizer for the biases.
                            (default :obj:`None`)
        trainable (bool): Whether or not to add the variables to the graph
                          collection ``tf.GraphKeys.TRAINABLE_VARIABLES``?
                          (default :obj:`True`)
        name (str): Optional name of this module
                    (argument of :class:`~tfsippet.scaffold.VarScopeObject`).
        scope (str): Optional scope of this module
                    (argument of :class:`~tfsippet.scaffold.VarScopeObject`).
    """

    def __init__(self,
                 num_outputs,
                 normalizer_fn=None,
                 normalizer_params=None,
                 weights_initializer=layers.xavier_initializer(),
                 weights_regularizer=None,
                 biases_initializer=tf.zeros_initializer(),
                 biases_regularizer=None,
                 trainable=True,
                 scope=None,
                 name=None):
        super(Linear, self).__init__(
            num_outputs=num_outputs,
            activation_fn=None,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            biases_initializer=biases_initializer,
            biases_regularizer=biases_regularizer,
            trainable=trainable,
            name=name,
            scope=scope,
        )
