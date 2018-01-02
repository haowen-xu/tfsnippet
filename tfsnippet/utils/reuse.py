import inspect
import functools
import weakref
from contextlib import contextmanager

import six
import tensorflow as tf

from .scope import reopen_variable_scope, root_variable_scope

__all__ = ['auto_reuse_variables', 'instance_reuse', 'global_reuse']


@contextmanager
def auto_reuse_variables(name_or_scope, reopen_name_scope=False, **kwargs):
    """
    Open a variable scope as a context, automatically choosing `reuse` flag.

    The `reuse` flag will be set to :obj:`False` if the variable scope is
    opened for the first time, and it will be set to :obj:`True` each time
    the variable scope is opened again.

    Args:
        name_or_scope (str or tf.VariableScope): The name of the variable
            scope, or the variable scope to open.
        reopen_name_scope (bool): Whether or not to re-open the original name
            scope of `name_or_scope`?  This option takes effect only if
            `name_or_scope` is actually an instance of
            :class:`tf.VariableScope`.
        \**kwargs: Named arguments for opening the variable scope.

    Yields:
        tf.VariableScope: The opened variable scope.
    """
    if not name_or_scope:
        raise ValueError('`name_or_scope` cannot be empty.  If you want to '
                         'auto-reuse variables in root variable scope, you '
                         'should capture the root variable scope instance '
                         'and call `auto_reuse_variables` on that, instead '
                         'of calling with an empty name')

    if reopen_name_scope:
        if not isinstance(name_or_scope, tf.VariableScope):
            raise ValueError('`reopen_name_scope` can be set to True '
                             'only if `name_or_scope` is an instance of '
                             '`tf.VariableScope`')

        def generate_context():
            return reopen_variable_scope(name_or_scope, **kwargs)
    else:
        def generate_context():
            return tf.variable_scope(name_or_scope, **kwargs)

    with generate_context() as vs:
        # check whether or not the variable scope has been initialized
        graph = tf.get_default_graph()
        if graph not in __auto_reuse_variables_graph_dict:
            __auto_reuse_variables_graph_dict[graph] = set([])
        initialized_scopes = __auto_reuse_variables_graph_dict[graph]
        reuse = vs.name in initialized_scopes

        # if `reuse` is True, set the reuse flag
        if reuse:
            vs.reuse_variables()
            yield vs
        else:
            yield vs
            initialized_scopes.add(vs.name)

#: dict to track the initialization state for each variable scope
#: belonging to every living graph.
__auto_reuse_variables_graph_dict = weakref.WeakKeyDictionary()


def instance_reuse(method=None, scope=None):
    """
    Decorate an instance method within :func:`auto_reuse_variables` context.

    This decorator should be applied to unbound instance methods, and
    the instance that owns the methods should have :attr:`variable_scope`
    attribute.  For example:

    .. code-block:: python

        class Foo(object):

            def __init__(self, name):
                with tf.variable_scope(name) as vs:
                    self.variable_scope = vs

            @instance_reuse
            def foo(self):
                return tf.get_variable('bar', ...)

    The above example is then equivalent to the following code:

    .. code-block:: python

        class Foo(object):

            def __init__(self, name):
                with tf.variable_scope(name) as vs:
                    self.variable_scope = vs

            def foo(self):
                with reopen_variable_scope(self.variable_scope):
                    with auto_reuse_variables('foo'):
                        return tf.get_variable('bar', ...)

    By default the name of the variable scope should be equal to the name
    of the decorated method, and the name scope within the context should
    be equal to the variable scope name, plus some suffix to make it unique.
    The variable scope name can be set by `scope` argument, for example:

    .. code-block:: python

        class Foo(object):

            @instance_reuse(scope='scope_name')
            def foo(self):
                return tf.get_variable('bar', ...)

    Note that the variable reusing is based on the name of the variable
    scope, rather than the method.  As a result, two methods with the same
    `scope` argument will reuse the same set of variables.  For example:

    .. code-block:: python

        class Foo(object):

            @instance_reuse(scope='foo')
            def foo_1(self):
                return tf.get_variable('bar', ...)

            @instance_reuse(scope='foo')
            def foo_2(self):
                return tf.get_variable('bar', ...)

    These two methods will return the same `bar` variable.

    Args:
        scope (str): The name of the variable scope. If not set, will use the
            method name as scope name. This argument must be specified as named
            argument.

    See Also:
        :func:`tfsnippet.utils.global_reuse`
    """
    if method is None:
        return functools.partial(instance_reuse, scope=scope)

    # check whether or not `method` looks like an instance method
    if six.PY2:
        getargspec = inspect.getargspec
    else:
        getargspec = inspect.getfullargspec

    if inspect.ismethod(method):
        raise TypeError('`method` is expected to be unbound instance method')
    argspec = getargspec(method)
    if not argspec.args or argspec.args[0] != 'self':
        raise TypeError('`method` seems not to be an instance method '
                        '(whose first argument should be `self`)')

    # determine the scope name
    scope = scope or method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        obj = args[0]
        variable_scope = obj.variable_scope
        if not isinstance(variable_scope, tf.VariableScope):
            raise TypeError('`variable_scope` attribute of the instance {!r} '
                            'is expected to be a `tf.VariableScope`, but got '
                            '{!r}'.format(obj, variable_scope))

        with reopen_variable_scope(variable_scope):
            with auto_reuse_variables(scope):
                return method(*args, **kwargs)

    return wrapper


def global_reuse(method=None, scope=None):
    """
    Decorate a function within :func:`auto_reuse_variables` scope globally.

    Any function or method applied with this decorator will be called within
    a variable scope opened first by :func:`root_variable_scope`, then by
    :func:`auto_reuse_variables`. That is to say, the following code:

    .. code-block:: python

        @global_reuse
        def foo():
            return tf.get_variable('bar', ...)

        bar = foo()

    is equivalent to:

    .. code-block:: python

        with root_variable_scope():
            with auto_reuse_variables('foo'):
                bar = tf.get_variable('bar', ...)

    By default the name of the variable scope should be equal to the name
    of the decorated method, and the name scope within the context should
    be equal to the variable scope name, plus some suffix to make it unique.
    The variable scope name can be set by `scope` argument, for example:

    .. code-block:: python

        @global_reuse(scope='dense')
        def dense_layer(inputs):
            w = tf.get_variable('w', ...)
            b = tf.get_variable('b', ...)
            return tf.matmul(w, inputs) + b

    Note that the variable reusing is based on the name of the variable
    scope, rather than the function object.  As a result, two functions
    with the same name, or with the same `scope` argument, will reuse
    the same set of variables.  For example:

    .. code-block:: python

        @global_reuse(scope='foo')
        def foo_1():
            return tf.get_variable('bar', ...)

        @global_reuse(scope='foo')
        def foo_2():
            return tf.get_variable('bar', ...)

    These two functions will return the same `bar` variable.

    Args:
        scope (str): The name of the variable scope. If not set, will use the
            function name as scope name. This argument must be specified as
            named argument.

    See Also:
        :func:`tfsnippet.utils.instance_reuse`
    """
    if method is None:
        return functools.partial(global_reuse, scope=scope)
    scope = scope or method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        with root_variable_scope():
            with auto_reuse_variables(scope):
                return method(*args, **kwargs)

    return wrapper
