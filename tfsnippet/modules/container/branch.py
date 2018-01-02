import re

import six
import tensorflow as tf

from ..base import Module

__all__ = ['ListMapper', 'DictMapper']

_VALID_KEY_FOR_DICT_MAPPER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


class ListMapper(Module):
    """
    Module that maps inputs into a list of outputs.

    This module is a branch module, which maps the inputs into a list
    of outputs, according to the specified `mapper` list.  Each item of
    `mapper` should be a function of a :class:`Module` instance, which
    produces corresponding output given the inputs.

    To merge the outputs of :class:`ListMapper`, one may use functions
    like :func:`tf.concat` or :func:`tf.stack`, e.g.:

    .. code-block:: python

        net = Sequential([
            ListMapper([...]),
            functools.partial(tf.concat, axis=0),
        ])

    in which the `net` module first maps the inputs into a list by using
    :class:`ListMapper`, then generates the final output by concatenating
    the outputs of :class:`ListMapper` along axis-0.
    """

    def __init__(self, mapper, name=None, scope=None):
        """
        Construct the :class:`ListMapper`.

        Args:
            mapper (list[(inputs, \**kwargs) -> outputs]): The mapper list.
            name (str): Optional name of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        mapper = tuple(mapper)
        if not mapper:
            raise ValueError('`mapper` must not be empty')

        super(ListMapper, self).__init__(name=name, scope=scope)
        self._mapper = mapper

    def _forward(self, inputs, **kwargs):
        ret = []
        for i, v in enumerate(self._mapper):
            with tf.variable_scope('_{}'.format(i)):
                ret.append(v(inputs, **kwargs))
        return ret


class DictMapper(Module):
    """
    Module that maps inputs into a dict of outputs.

    This module is a branch module, which maps the inputs into a dict
    of outputs, according to the specified `mapper` dict.  The keys of
    `mapper` should be valid Python identifiers, i.e., matching the
    pattern ``^[A-Za-z_][A-Za-z0-9_]*$``.  The values of `mapper` should
    be functions or :class:`Module` instances, which produces corresponding
    outputs given the inputs.

    A typical usage of :class:`DictMapper` is to derive the distribution
    parameters for a :class:`~tfsnippet.distributions.DistributionFactory`,
    e.g.:

    .. code-block:: python

        from tensorflow import keras as K

        from tfsnippet.distributions import Normal
        from tfsnippet.modules import Sequential, DictMapper

        net = Sequential([
            K.layers.Dense(100, activation=tf.nn.relu),
            DictMapper({
                'mean': K.layers.Dense(2),
                'logstd': K.layers.Dense(2),
            })
        ])
        factory = Normal.factory()
        distribution = factory(net(x))

    In the above example, the `net` module will produce a dict carrying `mean`
    and `logstd`, consumed by the `factory`.
    """

    def __init__(self, mapper, name=None, scope=None):
        """
        Construct the :class:`DictMapper`.

        Args:
            mapper (dict[str, (inputs, \**kwargs) -> outputs]): The mapper dict.
            name (str): Optional name of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        mapper = {k: v for k, v in six.iteritems(mapper)}
        if not mapper:
            raise ValueError('`mapper` must not be empty')
        for k in six.iterkeys(mapper):
            if not _VALID_KEY_FOR_DICT_MAPPER.match(k):
                raise ValueError('The key for `DictMapper` must be a valid '
                                 'Python identifier (matching the pattern '
                                 '"^[A-Za-z_][A-Za-z0-9_]*$")')

        super(DictMapper, self).__init__(name=name, scope=scope)
        self._mapper = mapper

    def _forward(self, inputs, **kwargs):
        ret = {}
        for k, v in six.iteritems(self._mapper):
            with tf.variable_scope(k):
                ret[k] = v(inputs, **kwargs)
        return ret
