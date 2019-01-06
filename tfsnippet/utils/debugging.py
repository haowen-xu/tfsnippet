import os
from contextlib import contextmanager

import tensorflow as tf

from .doc_utils import add_name_arg_doc

__all__ = [
    'is_assertion_enabled',
    'set_assertion_enabled',
    'scoped_set_assertion_enabled',
    'should_check_numerics',
    'set_check_numerics',
    'scoped_set_check_numerics',
    'maybe_check_numerics',
    'assert_deps',
]


_enable_assertion = (
    os.environ.get('TFSNIPPET_DISABLE_ASSERTION', '').lower()
    not in ('1', 'yes', 'on', 'true')
)
_check_numerics = (
    os.environ.get('TFSNIPPET_CHECK_NUMERICS', '').lower()
    in ('1', 'yes', 'on', 'true')
)


@contextmanager
def _scoped_set(value, getter, setter):
    old_value = getter()
    try:
        setter(value)
        yield
    finally:
        setter(old_value)


def is_assertion_enabled():
    """Whether or not to enable assertions?"""
    return _enable_assertion


def set_assertion_enabled(enabled):
    """
    Set whether or not to enable assertions?

    If the assertions are disabled, then :func:`assert_deps` will not execute
    any given operations.
    """
    global _enable_assertion
    _enable_assertion = bool(enabled)


def scoped_set_assertion_enabled(enabled):
    """Set whether or not to enable assertions in a scoped context."""
    return _scoped_set(enabled, is_assertion_enabled, set_assertion_enabled)


def should_check_numerics():
    """Whether or not to check numerics?"""
    return _check_numerics


def set_check_numerics(enabled):
    """
    Set whether or not to check numerics?

    By checking numerics, one can figure out where the NaNs and Infinities
    originate from.  This affects the behavior of :func:`maybe_check_numerics`,
    and the default behavior of :class:`tfsnippet.distributions.Distribution`
    sub-classes.
    """
    global _check_numerics
    _check_numerics = bool(enabled)


def scoped_set_check_numerics(enabled):
    """Set whether or not to check numerics in a scoped context."""
    return _scoped_set(enabled, should_check_numerics, set_check_numerics)


@add_name_arg_doc
def maybe_check_numerics(tensor, message, name=None):
    """
    Check the numerics of `tensor`, if ``should_check_numerics()``.

    Args:
        tensor: The tensor to be checked.
        message: The message to display when numerical issues occur.

    Returns:
        tf.Tensor: The tensor, whose numerics have been checked.
    """
    if should_check_numerics():
        return tf.check_numerics(tensor, message, name=name)
    else:
        return tf.identity(tensor)


@contextmanager
def assert_deps(assert_ops):
    """
    If ``is_assertion_enabled() == True``, open a context that will run
    `assert_ops` on exit.  Otherwise do nothing.

    Args:
        assert_ops (Iterable[tf.Operation or NOne]): A list of assertion
            operations.  :obj:`None` items will be ignored.

    Yields:
        bool: A boolean indicate whether or not the assertion operations
            are not empty, and are executed.
    """
    assert_ops = [o for o in assert_ops if o is not None]
    if assert_ops and is_assertion_enabled():
        with tf.control_dependencies(assert_ops):
            yield True
    else:
        yield False
