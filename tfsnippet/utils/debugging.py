import warnings
from contextlib import contextmanager

import tensorflow as tf

__all__ = [
    'is_assertion_enabled',
    'set_assertion_enabled',
    'scoped_set_assertion_enabled',
    'maybe_assert',
    'control_deps',
    'should_check_numerics',
    'set_check_numerics',
    'scoped_set_check_numerics',
    'maybe_check_numerics',
    'check_numerics',
]


_enable_assertion = True
_check_numerics = False


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

    If the assertions are disabled, all the assertions statements in
    tfsnippet are disabled.  User assertions created with the helper
    function :func:`maybe_assert` are also affected by this option.
    """
    global _enable_assertion
    _enable_assertion = bool(enabled)


def scoped_set_assertion_enabled(enabled):
    """Set whether or not to enable assertions in a scoped context."""
    return _scoped_set(enabled, is_assertion_enabled, set_assertion_enabled)


def maybe_assert(assert_fn, *args, **kwargs):
    """
    Maybe call `assert_fn` to generate an assertion operation.

    The assertion will execute only if ``is_assertions_enabled() == True``.

    Args:
        assert_fn: The assertion function, e.g.,
            :func:`tensorflow.assert_rank_at_least`.
        *args: The arguments to be passed to `assert_fn`.
        \**kwargs: The named arguments to be passed to `assert_fn`.

    Returns:
        tf.Operation or None: The assertion operation if the assertion is
            enabled, or :obj:`None` otherwise.
    """
    if is_assertion_enabled():
        return assert_fn(*args, **kwargs)


@contextmanager
def control_deps(control_inputs):
    """
    A wrapper of :func:`tensorflow.control_dependencies`, where the :obj:`None`
    in specified `control_inputs` are filtered out.

    Args:
        control_inputs (Iterable[tf.Operation or None]): The operations to
            be executed, or :obj:`None`.
    """
    control_inputs = [o for o in control_inputs if o is not None]
    if control_inputs:
        with tf.control_dependencies(control_inputs) as r:
            yield r
    else:
        yield


def should_check_numerics():
    """Whether or not to check numerics?"""
    return _check_numerics


def set_check_numerics(enabled):
    """
    Set whether or not to check numerics?

    By checking numerics, one can figure out where the NaNs and Infinities
    originate from.  This option affects the behavior of :func:`check_numerics`,
    and the default behavior of :class:`tfsnippet.distributions.Distribution`
    sub-classes.
    """
    global _check_numerics
    _check_numerics = bool(enabled)


def scoped_set_check_numerics(enabled):
    """Set whether or not to check numerics in a scoped context."""
    return _scoped_set(enabled, should_check_numerics, set_check_numerics)


def maybe_check_numerics(tensor, message, name=None):
    """
    Check the numerics of `tensor`, if ``should_check_numerics()``.

    Args:
        tensor: The tensor to be checked.
        message: The message to display when numerical issues occur.
        name: Optional name of the check op.

    Returns:
        tf.Tensor: The tensor, whose numerics have been checked.
    """
    if should_check_numerics():
        return tf.check_numerics(tensor, message, name=name)
    else:
        return tf.identity(tensor)


def check_numerics(*args, **kwargs):  # pragma: no cover
    warnings.warn('`check_numerics` has been deprecated, use '
                  '`maybe_check_numerics` instead.',
                  category=DeprecationWarning)
    return maybe_check_numerics(*args, **kwargs)
