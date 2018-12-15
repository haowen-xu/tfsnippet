import numpy as np

from tfsnippet.mathops import npyops, tfops
from tfsnippet.utils import get_default_session_or_error

__all__ = ['run_op', 'assert_op_result']


def run_op(ops, op, *args, **kwargs):
    ret = op(ops, *args, **kwargs)
    if ops is tfops:
        ret = get_default_session_or_error().run(ret)
    return ret


def assert_op_result(expected, op, *args, **kwargs):
    for ops in [npyops, tfops]:
        np.testing.assert_allclose(
            expected,
            run_op(ops, op, *args, **kwargs)
        )


def naive_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

