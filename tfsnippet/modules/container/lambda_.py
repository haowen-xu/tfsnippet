from ..base import Module

__all__ = ['Lambda']


class Lambda(Module):
    """
    Wrapping arbitrary function into a neural network :class:`Module`.

    This class wraps an arbitrary function or lambda expression into
    a neural network :class:`Module`, reusing the variables created
    within the specified function.

    For example, one may wrap :func:`tensorflow.contrib.layers.fully_connected`
    into a reusable module with :class:`Lambda` component as follows:

    .. code-block:: python

        import functools
        from tensorflow.contrib import layers

        dense = Lambda(
            functools.partial(
                layers.fully_connected,
                num_outputs=100,
                activation_fn=tf.nn.relu
            )
        )
    """

    def __init__(self, f, name=None, scope=None):
        """
        Construct the :class:`Lambda`.

        Args:
            f ((inputs, \**kwargs) -> outputs): The function or lambda
                expression which derives the outputs.
            name (str): Optional name of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        super(Lambda, self).__init__(name=name, scope=scope)
        self._factory = f

    def _forward(self, inputs, **kwargs):
        return self._factory(inputs, **kwargs)
