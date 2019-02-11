from contextlib import contextmanager

import tensorflow as tf

from .doc_utils import add_name_arg_doc
from .graph_keys import GraphKeys

__all__ = [
    'maybe_check_numerics',
    'assert_deps',
    'maybe_add_histogram',
]


@add_name_arg_doc
def maybe_check_numerics(tensor, message, name=None):
    """
    If ``tfsnippet.settings.check_numerics == True``, check the numerics of
    `tensor`.  Otherwise do nothing.

    Args:
        tensor: The tensor to be checked.
        message: The message to display when numerical issues occur.

    Returns:
        tf.Tensor: The tensor, whose numerics have been checked.
    """
    from .settings_ import settings
    if settings.check_numerics:
        tensor = tf.convert_to_tensor(tensor)
        return tf.check_numerics(tensor, message, name=name)
    else:
        return tensor


@contextmanager
def assert_deps(assert_ops):
    """
    If ``tfsnippet.settings.enable_assertions == True``, open a context that
    will run `assert_ops`.  Otherwise do nothing.

    Args:
        assert_ops (Iterable[tf.Operation or None]): A list of assertion
            operations.  :obj:`None` items will be ignored.

    Yields:
        bool: A boolean indicate whether or not the assertion operations
            are not empty, and are executed.
    """
    from .settings_ import settings
    assert_ops = [o for o in assert_ops if o is not None]
    if assert_ops and settings.enable_assertions:
        with tf.control_dependencies(assert_ops):
            yield True
    else:
        for op in assert_ops:
            # let TensorFlow not warn about not using this assertion operation
            if hasattr(op, 'mark_used'):
                op.mark_used()
        yield False


@add_name_arg_doc
def maybe_add_histogram(tensor, summary_name=None, strip_scope=False,
                        collections=None, name=None):
    """
    If ``tfsnippet.settings.auto_histogram == True``, add the histogram
    of `tensor` via :func:`tfsnippet.add_histogram`.  Otherwise do nothing.

    Args:
        tensor: Take histogram of this tensor.
        summary_name: Specify the summary name for `tensor`.
        strip_scope: If :obj:`True`, strip the name scope from `tensor.name`
            when adding the histogram.
        collections: Add the histogram to these collections. Defaults to
            `[tfsnippet.GraphKeys.AUTO_HISTOGRAM]`.

    Returns:
        The serialized histogram tensor of `tensor`.

    See Also:
        :func:`tfsnippet.add_histogram`
    """
    from .settings_ import settings
    from .summary_collector import add_histogram
    if settings.auto_histogram:
        if collections is None:
            collections = (GraphKeys.AUTO_HISTOGRAM,)
        return add_histogram(
            tensor, summary_name=summary_name, collections=collections,
            strip_scope=strip_scope, name=name or 'maybe_add_histogram'
        )
