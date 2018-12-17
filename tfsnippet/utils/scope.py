from contextlib import contextmanager

import six
import tensorflow as tf
from tensorflow.python.ops import variable_scope as variable_scope_ops

from .doc_inherit import DocInherit
from .misc import camel_to_underscore

__all__ = [
    'get_valid_name_scope_name',
    'reopen_variable_scope',
    'root_variable_scope',
    'VarScopeObject'
]


def get_valid_name_scope_name(name, cls_or_instance=None):
    """
    Generate a valid name scope name.

    Args:
        name (str): The base name.
        cls_or_instance: The class or the instance object, optional.

    Returns:
        str: The generated scope name.
    """
    # TODO: add more validation here.
    prefix = ''
    if cls_or_instance is not None:
        if not isinstance(cls_or_instance, six.class_types):
            cls_or_instance = cls_or_instance.__class__
        prefix = '{}.'.format(cls_or_instance.__name__).lstrip('_')
    return prefix + name


@contextmanager
def reopen_variable_scope(var_scope, **kwargs):
    """
    Reopen the specified `var_scope` and its original name scope.

    Args:
        var_scope (tf.VariableScope): The variable scope instance.
        **kwargs: Named arguments for opening the variable scope.
    """
    if not isinstance(var_scope, tf.VariableScope):
        raise TypeError('`var_scope` must be an instance of `tf.VariableScope`')

    with tf.variable_scope(var_scope,
                           auxiliary_name_scope=False,
                           **kwargs) as vs:
        with tf.name_scope(var_scope.original_name_scope):
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


@DocInherit
class VarScopeObject(object):
    """
    Base class for objects that own a variable scope.

    The :class:`VarScopeObject` can be used along with :func:`instance_reuse`,
    for example::

        class YourVarScopeObject(VarScopeObject):

            @instance_reuse
            def foo(self):
                return tf.get_variable('bar', ...)

        o = YourVarScopeObject('object_name')
        o.foo()  # You should get a variable with name "object_name/foo/bar"
    """

    def __init__(self, name=None, scope=None):
        """
        Construct the :class:`VarScopeObject`.

        Args:
            name (str): Name of this object.  A unique variable scope name
                would be picked up according to this argument, if `scope` is
                not specified.  If both this argument and `scope` is not
                specified, the underscored class name would be considered as
                `name`.  This argument will be stored and can be accessed via
                :attr:`name` attribute of the instance.  If not specified,
                :attr:`name` would be :obj:`None`.
            scope (str): Scope of this object.  If specified, it will be used
                as the variable scope name, even if another object has already
                taken the same scope.  That is to say, these two objects will
                share the same variable scope.
        """
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
            self._variable_scope_created(vs)

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.variable_scope.name)

    def _variable_scope_created(self, vs):
        """
        Derived classes may override this to execute code right after
        the variable scope has been created.

        Args:
            vs (tf.VariableScope): The created variable scope.
        """

    @property
    def name(self):
        """Get the name of this object."""
        return self._name

    @property
    def variable_scope(self):
        """Get the variable scope of this object."""
        return self._variable_scope
