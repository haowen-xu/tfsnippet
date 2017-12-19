import tensorflow as tf

from tfsnippet.utils import VarScopeObject, auto_reuse_variables

__all__ = ['Module']


class Module(VarScopeObject):
    """
    Base class for neural network modules.

    A neural network module is basically a reusable object that could derive
    outputs for inputs, using the same, fixed set of parameters every time.
    This class provides a base for implementing such a reusable module, which
    adopts :class:`VarScopeObject` and :func:`auto_reuse_variables` to achieve
    parameter sharing, making it possible to use all other 3rd party libraries
    with this module interface of TFSnippet.

    For example, a reusable dense layer may be implemented as follows:

    .. code-block:: python

        class Dense(Module):

            def __init__(self, num_outputs, activation_fn, **kwargs):
                super(Dense, self).__init__(**kwargs)
                self.num_outputs = num_outputs
                self.activation_fn = activation_fn

            def _forward(self, inputs, **kwargs):
                return tf.contrib.layers.fully_connected(
                    inputs,
                    activation_fn=self.activation_fn,
                    num_outputs=self.num_outputs
                )

        dense = Dense(num_outputs)
        y1 = dense(x1)
        y2 = dense(x2)  # parameters shared with y1

    It is even possible to construct module instances within :meth:`_forward`
    of an already established module instance, with parameter sharing working
    properly, for example:

    .. code-block:: python

        class ConstantBiasedSoftPlus(Module):

            def __init__(self, num_outputs, biases, **kwargs):
                super(ConstantBiasedSoftPlus, self).__init__(**kwargs)
                self.num_outputs = num_outputs
                self.biases = biases

            def _forward(self, inputs, **kwargs):
                softplus = Dense(self.num_outputs, activation_fn=tf.nn.softplus)
                outputs = softplus(inputs)
                return outputs + self.biases
    """

    def _forward(self, inputs, **kwargs):
        """
        Derive outputs for specified `inputs`.

        Overriding this in a child class to actually implement the module.
        This method is not designed for calling directly.  Instead, it is
        designed to be called from :meth:`__call__`, where the variable
        scope ``self.variable_scope + 'forward'`` is opened, with reusing
        flag set automatically.

        Args:
            inputs: The input tensor(s).
            **kwargs: Named arguments for deriving the outputs.

        Returns:
            The derived outputs.
        """
        raise NotImplementedError()

    def __call__(self, inputs, **kwargs):
        with auto_reuse_variables(self.variable_scope,
                                  reopen_name_scope=True):
            # Here `reopen_name_scope` is set to True, so that multiple
            # calls to the same Module instance will always generate operations
            # within the original name scope.
            # However, in order for ``tf.variable_scope(default_name=...)``
            # to work properly with variable reusing, we must generate a nested
            # unique name scope.
            with tf.name_scope('forward'):
                return self._forward(inputs, **kwargs)
