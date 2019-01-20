import tensorflow as tf

__all__ = ['assert_variables']


def assert_variables(names, exist=True, trainable=None, scope=None,
                     collections=None):
    """
    Assert variables of `name_or_names` meet certain criterion.

    Args:
        names (Iterable[str]): Name, or names.
        exist (bool): Assert variables exist or not.
        trainable: Assert variables are trainable or not.
        scope (None or str): The scope prefix to be prepended to the names.
        collections (Iterable[str]): Additional graph collections, where
            to ensure the variables are in.
    """
    def normalize_name(n):
        return n.rsplit(':', 1)[0]

    names = tuple(names)

    if scope:
        scope = str(scope).rstrip('/')
        names = tuple('{}/{}'.format(scope, name) for name in names)

    global_vars = {normalize_name(v.name): v
                   for v in tf.global_variables()}
    trainable_vars = {normalize_name(v.name): v
                      for v in tf.trainable_variables()}
    collections = list(collections or ())
    collection_vars = [
        {normalize_name(v.name): v for v in tf.get_collection(c)}
        for c in collections
    ]

    for name in names:
        if exist:
            if name not in global_vars:
                raise AssertionError('Variable `{}` is expected to exist, but '
                                     'turn out to be non-exist.'.format(name))

            # check trainable
            if trainable is False:
                if name in trainable_vars:
                    raise AssertionError('Variable `{}` is expected not to be '
                                         'trainable, but turned out to be '
                                         'trainable'.format(name))
            elif trainable is True:
                if name not in trainable_vars:
                    raise AssertionError('Variable `{}` is expected to be '
                                         'trainable, but turned out not to be '
                                         'trainable'.format(name))

            # check collections
            for coll, coll_vars in zip(collections, collection_vars):
                if name not in coll_vars:
                    raise AssertionError('Variable `{}` is expected to be '
                                         'in the collection `{}`, but turned '
                                         'out not.'.format(name, coll))

        else:
            if name in global_vars:
                raise AssertionError('Variable `{}` is expected not to exist, '
                                     'but turn out to be exist.'.format(name))
