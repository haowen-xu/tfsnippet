import six
import tensorflow as tf

__all__ = [
    'graph_collections_snapshot',
    'GraphCollectionsDiff',
    'diff_graph_collections',
    'add_name_scope',
    'add_variable_scope',
]


def graph_collections_snapshot(graph=None, keys=None):
    """
    Take a snapshot of the graph collections.

    Args:
        graph (None or tf.Graph): The graph object.  If not specified,
            use the current default graph.
        keys (Iterable[str]): If specified, only take the snapshot on
            these collection keys.  Otherwise take the snapshot on
            all collection keys.

    Returns:
        dict[str, list]: The snapshot of collections.
    """
    if keys is not None:
        keys = set(keys)
    g = graph or tf.get_default_graph()
    return {
        k: list(g.get_collection_ref(k))
        for k in g.get_all_collection_keys()
        if keys is None or k in keys
    }


class GraphCollectionsDiff(object):
    """Class to hold the result of :func:`diff_collections`."""

    def __init__(self, added, removed):
        """
        Construct a new :class:`GraphCollectionsDiff`.

        Args:
            added (dict[str, list]): The objects added to each collection.
            removed (dict[str, list]): The objects removed from each collection.
        """
        self.added = added
        self.removed = removed


def diff_graph_collections(old, new):
    """
    Compute the difference between the `old` and `new` collections snapshot.

    Args:
        old (dict[str, list]): The old snapshot of graph collections.
        new (dict[str, list]): The new snapshot of graph collections.

    Returns:
        GraphCollectionsDiff: The diff result.
    """
    def partial_diff(a, b):
        ret = {}
        for k, c in six.iteritems(a):
            if k not in b:
                ret[k] = c
            else:
                ret[k] = list(set(c).difference(b[k]))
        return ret

    return GraphCollectionsDiff(added=partial_diff(new, old),
                                removed=partial_diff(old, new))


def add_name_scope(method):
    """
    Automatically open a new name scope when calling the method.

    Usage::

        @add_name_scope
        def dense(inputs, name=None):
            return tf.layers.dense(inputs)

    Args:
        method: The method to decorate.  It must accept an optional named
            argument `name`, to receive the inbound name argument.
            If the `name` argument is not specified as named argument during
            calling, the name of the method will be used as `name`.

    Returns:
        The decorated method.
    """
    method_name = method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        if kwargs.get('name') is None:
            kwargs['name'] = method_name
        with tf.name_scope(kwargs['name']):
            return method(*args, **kwargs)
    return wrapper


def add_variable_scope(method):
    """
    Automatically open a new variable scope when calling the method.

    Usage::

        @add_variable_scope
        def dense(inputs):
            return tf.layers.dense(inputs)

    Args:
        method: The method to decorate.
            If the `name` argument is not specified as named argument during
            calling, the name of the method will be used as `name`.

    Returns:
        The decorated method.
    """
    method_name = method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        name = kwargs.pop('name', None)
        with tf.variable_scope(name, default_name=method_name):
            return method(*args, **kwargs)
    return wrapper
