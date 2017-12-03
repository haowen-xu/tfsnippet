# -*- coding: utf-8 -*-
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.python.ops import variable_scope as variable_scope_ops

from tfsnippet.utils import camel_to_underscore

__all__ = ['reopen_variable_scope', 'root_variable_scope', 'VarScopeObject']


@contextmanager
def reopen_variable_scope(var_scope, **kwargs):
    """
    Reopen the specified `var_scope` and its original name scope.

    Unlike :func:`tf.variable_scope`, which does not open the original name
    scope even if a stored :class:`tf.VariableScope` instance is specified,
    this method opens exactly the same name scope as the original one.

    Args:
        var_scope (tf.VariableScope): The variable scope instance.
        **kwargs: Named arguments for opening the variable scope.
    """
    if not isinstance(var_scope, tf.VariableScope):
        raise TypeError('`var_scope` must be an instance of `tf.VariableScope`')
    old_name_scope = var_scope.original_name_scope
    with variable_scope_ops._pure_variable_scope(var_scope, **kwargs) as vs:
        name_scope = old_name_scope
        if name_scope and not name_scope.endswith('/'):
            name_scope += '/'

        with tf.name_scope(name_scope):
            yield vs


@contextmanager
def root_variable_scope(**kwargs):
    """
    Open the root variable scope and its name scope.

    Args:
        **kwargs: Named arguments for opening the root variable scope.
    """
    # `tf.variable_scope` does not support opening the root variable scope
    # from empty name.  It always prepend the name of current variable scope
    # to the front of opened variable scope.  So we get the current scope,
    # and pretend it to be the root scope.
    scope = tf.get_variable_scope()
    old_name = scope.name
    try:
        scope._name = ''
        with variable_scope_ops._pure_variable_scope('', **kwargs) as vs:
            scope._name = old_name
            with tf.name_scope(None):
                yield vs
    finally:
        scope._name = old_name


class VarScopeObject(object):
    """
    Base class for object that owns a variable scope.

    It is typically used along with :func:`tfsnippet.scaffold.instance_reuse`.

    Args:
        name (str):
            Name of this object.

            A unique variable scope name would be picked up according to this
            argument, if `scope` is not specified.
            If both this argument and `scope` is not specified, the underscored
            class name would be considered as `name`.

            This argument will be stored and can be accessed via :attr:`name`
            attribute of the instance.  If not specified, :attr:`name` would
            be :obj:`None`.

        scope (str):
            Scope of this object.

            If specified, it will be used as the variable scope name, even if
            another object has already taken the same scope.  That is to say,
            these two objects will share the same variable scope.
    """

    def __init__(self, name=None, scope=None):
        scope = scope or None
        name = name or None

        if not scope and not name:
            default_name = camel_to_underscore(self.__class__.__name__)
            default_name = default_name.lstrip('_')
        else:
            default_name = name

        with tf.variable_scope(scope, default_name=default_name) as vs:
            self._variable_scope = vs       # type: tf.VariableScope
            self._name = name

    @property
    def name(self):
        """Get the name of this object."""
        return self._name

    @property
    def variable_scope(self):
        """Get the variable scope of this object."""
        return self._variable_scope

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.variable_scope.name)
