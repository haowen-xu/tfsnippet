import functools
import weakref

import six
import tensorflow as tf

from .scope import root_variable_scope
from .tfver import is_tensorflow_version_higher_or_equal

__all__ = ['global_reuse']


def require_at_least_tensorflow_1_5():
    if not is_tensorflow_version_higher_or_equal('1.5.0'):  # pragma: no cover
        raise RuntimeError('The reuse utilities are only tested for '
                           'TensorFlow >= 1.5.0.  Using these utilities with '
                           'any lower versions of TensorFlow are totally '
                           'not allowed.')


def global_reuse(method_or_scope=None, _sentinel=None, scope=None):
    """
    Decorate a function to reuse a variable scope automatically.

    The first time to enter a function decorated by this utility will
    open a new variable scope under the root variable scope.
    This variable scope will be reused the next time to enter this function.
    For example::

    .. code-block:: python

        @global_reuse
        def foo():
            return tf.get_variable('bar', ...)

        bar = foo()
        bar_2 = foo()
        assert(bar is bar_2)  # should be True

    By default the name of the variable scope should be chosen according to
    the name of the decorated method.  You can change this behavior by
    specifying an alternative name, for example:

    .. code-block:: python

        @global_reuse('dense')
        def dense_layer(inputs):
            w = tf.get_variable('w', ...)  # name will be 'dense/w'
            b = tf.get_variable('b', ...)  # name will be 'dense/b'
            return tf.matmul(w, inputs) + b

    If you have two functions sharing the same scope name, they will not
    use the same variable scope.  Instead, one of these two functions will
    have its scope name added with a suffix '_1', for example::

    .. code-block:: python

        @global_reuse('foo')
        def foo_1():
            return tf.get_variable('bar', ...)

        @global_reuse('foo')
        def foo_2():
            return tf.get_variable('bar', ...)

        assert(foo_1().name == 'foo/bar')
        assert(foo_2().name == 'foo_1/bar')

    The variable scope name will depend on the calling order of these
    two functions, so you should better not guess the scope name by yourself.

    Args:
        scope (str): The name of the variable scope. If not set, will use the
            function name as scope name. This argument must be specified as
            named argument.
    """
    require_at_least_tensorflow_1_5()

    if _sentinel is not None:  # pragma: no cover
        raise TypeError('`scope` must be specified with named argument.')

    if isinstance(method_or_scope, six.string_types):
        scope = method_or_scope
        method = None
    else:
        method = method_or_scope

    if method is None:
        return functools.partial(global_reuse, scope=scope)

    scope = scope or method.__name__

    if '/' in scope:
        raise ValueError('`global_reuse` does not support "/" in scope name.')

    # Until now, we have checked all the arguments, such that `method`
    # is the function to be decorated, and `scope` is the base name
    # for the variable scope.  We can now generate the closure used
    # to track the variable scopes.
    variable_scopes = weakref.WeakKeyDictionary()

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        graph = tf.get_default_graph()

        if graph not in variable_scopes:
            # Branch #1.1: first time to enter the function, and we are not
            #   in the root variable scope.  We should pick up the root
            #   variable scope before creating our desired variable scope.
            #   However, if we execute the method right after we obtain the
            #   new variable scope, we will not be in the correct name scope.
            #   So we should exit the root scope, then re-enter our desired
            #   variable scope.
            if tf.get_variable_scope().name:
                with root_variable_scope():
                    with tf.variable_scope(None, default_name=scope) as vs:
                        variable_scopes[graph] = vs
                with tf.variable_scope(vs):
                    return method(*args, **kwargs)

            # Branch #1.2: first time to enter the function, and we are just
            #   in the root variable scope.  So we can happily create a new
            #   variable scope, and just call the method immediately.
            else:
                with tf.variable_scope(None, default_name=scope) as vs:
                    variable_scopes[graph] = vs
                    return method(*args, **kwargs)

        else:
            # Branch #2: not the first time to enter the function, so we
            #   should reopen the variable scope with reuse set to `True`.
            vs = variable_scopes[graph]
            with tf.variable_scope(vs, reuse=True):
                return method(*args, **kwargs)

    return wrapper
