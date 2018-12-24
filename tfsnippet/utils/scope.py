import functools
from contextlib import contextmanager

import six
import tensorflow as tf
from tensorflow.python.ops import variable_scope as variable_scope_ops

from .doc_utils import DocInherit
from .misc import camel_to_underscore

__all__ = [
    'get_default_scope_name',
    'add_name_scope',
    'reopen_variable_scope',
    'root_variable_scope',
    'VarScopeObject'
]


def get_default_scope_name(name, cls_or_instance=None):
    """
    Generate a valid default scope name.

    Args:
        name (str): The base name.
        cls_or_instance: The class or the instance object, optional.
            If it has attribute ``variable_scope``, then ``variable_scope.name``
            will be used as a hint for the name prefix.  Otherwise, its class
            name will be used as the name prefix.

    Returns:
        str: The generated scope name.
    """
    # compose the candidate name
    prefix = ''
    if cls_or_instance is not None:
        if hasattr(cls_or_instance, 'variable_scope') and \
                isinstance(cls_or_instance.variable_scope, tf.VariableScope):
            vs_name = cls_or_instance.variable_scope.name
            vs_name = vs_name.rsplit('/', 1)[-1]
            prefix = '{}.'.format(vs_name)
        else:
            if not isinstance(cls_or_instance, six.class_types):
                cls_or_instance = cls_or_instance.__class__
            prefix = '{}.'.format(cls_or_instance.__name__).lstrip('_')
    name = prefix + name

    # validate the name
    name = name.lstrip('_')
    return name


def add_name_scope(method_or_name, _sentinel=None, default_name=None):
    if _sentinel is not None:  # pragma: no cover
        raise TypeError('`default_name` must be specified as named argument.')

    if isinstance(method_or_name, six.string_types):
        default_name = method_or_name
        method = None
    else:
        method = method_or_name

    if method is None:
        return functools.partial(add_name_scope, name=default_name)

    default_name = default_name or method.__name__
    if '/' in default_name:
        raise ValueError('`add_name_scope` does not support "/" in scope name.')

    # Until now, we have checked all the arguments, such that `method`
    # is the function to be decorated, and `name` is the default name
    # for the variable scope.
    @six.wraps(method)
    def wrapped(*args, **kwargs):
        name = kwargs.pop('name', None)
        scope = kwargs.pop('scope', None)

        with tf.name_scope(scope, default_name=name or default_name):
            return method(*args, **kwargs)

    return wrapped


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

    To build variables in the constructor of derived classes, you may use
    ``reopen_variable_scope(self.variable_scope)`` to open the original
    variable scope and its name scope, right after the constructor of
    :class:`VarScopeObject` has been called, for example::

        class YourVarScopeObject(VarScopeObject):

            def __init__(self, name=None, scope=None):
                super(YourVarScopeObject, self).__init__(name=name, scope=scope)
                with reopen_variable_scope(self.variable_scope):
                    self.w = tf.get_variable('w', ...)

    See Also:
        :func:`tfsnippet.utils.instance_reuse`.
    """

    def __init__(self, name=None, scope=None):
        """
        Construct the :class:`VarScopeObject`.

        Args:
            name (str): Default name of the variable scope.  Will be uniquified.
                If not specified, generate one according to the class name.
            scope (str): The name of the variable scope.
        """
        scope = scope or None
        name = name or None

        if not scope and not name:
            default_name = get_default_scope_name(
                camel_to_underscore(self.__class__.__name__))
        else:
            default_name = name

        with tf.variable_scope(scope, default_name=default_name) as vs:
            self._variable_scope = vs       # type: tf.VariableScope
            self._name = name

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.variable_scope.name)

    @property
    def name(self):
        """Get the name of this object."""
        return self._name

    @property
    def variable_scope(self):
        """Get the variable scope of this object."""
        return self._variable_scope
