from contextlib import contextmanager

import numpy as np


@contextmanager
def name_scope(name, default_name=None, values=None):
    """Dummy method for having identical interface with tfops."""
    yield


def ensure_axis_arg(axis):
    """
    Ensure the type of ``axis`` can be accepted by other math operations.
    """
    return tuple(np.asarray(axis, dtype=np.int32).tolist())


int32 = np.int32
identity = lambda x: x

abs = np.abs
sign = np.sign
log = np.log
log1p = np.log1p
exp = np.exp
clip_by_value = np.clip

convert_to_tensor = np.asarray
reduce_mean = np.mean
reduce_sum = np.sum
reduce_max = np.max
reduce_min = np.min
squeeze = np.squeeze

range = np.arange
shape = np.shape
rank = np.ndim
reshape = np.reshape


@contextmanager
def assert_rank_at_least(x, expected_rank, message=None):
    """
    Assert the rank of `x` is at least `expected_rank`.

    Args:
        x: The array to be asserted.
        expected_rank: The expected rank of `x`.
        message: The error message to display when assertion fails.

    Yields:
        A context, any operation defined within which should trigger assertion.
    """
    rank = np.ndim(x)
    if rank >= expected_rank:
        yield
    else:
        tag = 'rank(x): expected >= {!r}, got {!r}'.format(expected_rank, rank)
        if message:
            message = '{}: {}'.format(tag, message)
        else:
            message = tag
        raise AssertionError(message)
