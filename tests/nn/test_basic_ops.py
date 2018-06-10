import unittest

import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.nn import npyops, tfops
from .utils import run_op, assert_op_result


def wrap_op(op_name, *args, **kwargs):
    def wrapper(ops, *args2, **kwargs2):
        m_args = args + args2
        m_kwargs = {}
        m_kwargs.update(kwargs)
        m_kwargs.update(kwargs2)
        return getattr(ops, op_name)(*m_args, **m_kwargs)
    return wrapper


class BasicOpsTestCase(tf.test.TestCase):
    """Test whether or not the arguments of ops are expected."""

    def test_name_scope(self):
        with self.test_session():
            for ops in [npyops, tfops]:
                a = ops.convert_to_tensor(np.asarray([1, 2, 3]))
                with ops.name_scope(
                        'name_scope',
                        default_name='the_name_scope',
                        values=[a]):
                    np.testing.assert_equal(
                        [2, 3, 4],
                        run_op(ops, lambda ops: a + 1)
                    )

    def test_ensure_axis_arg(self):
        with self.test_session():
            for ops in [npyops, tfops]:
                axis = ops.ensure_axis_arg(ops.range(3, dtype=ops.int32))
                a = ops.reshape(ops.range(24), [2, 2, 2, 3])
                np.testing.assert_equal(
                    np.sum(np.arange(24).reshape([2, 2, 2, 3]), axis=(0, 1, 2)),
                    run_op(ops, wrap_op('reduce_sum'), a, axis=axis)
                )

    def test_many_op(self):
        with self.test_session():
            x = np.linspace(0, 1, 101)
            assert_op_result(
                x,
                wrap_op('identity', x)
            )
            assert_op_result(
                np.log(x),
                wrap_op('log', x)
            )
            assert_op_result(
                np.log1p(x),
                wrap_op('log1p', x)
            )
            assert_op_result(
                np.exp(x),
                wrap_op('exp', x)
            )
            assert_op_result(
                np.clip(x, .1, .95),
                wrap_op('clip_by_value', x, .1, .95)
            )

    def test_reduce_sum(self):
        with self.test_session():
            x = np.arange(24).reshape([2, 2, 2, 3])
            assert_op_result(
                np.sum(x, axis=-1, keepdims=False),
                wrap_op('reduce_sum', x, axis=-1)
            )
            assert_op_result(
                np.sum(x, axis=(0, 2), keepdims=False),
                wrap_op('reduce_sum', x, keepdims=False, axis=(0, 2))
            )
            assert_op_result(
                np.sum(x, axis=(0, 2), keepdims=True),
                wrap_op('reduce_sum', x, keepdims=True, axis=(0, 2))
            )
            assert_op_result(
                np.sum(x, axis=None, keepdims=False),
                wrap_op('reduce_sum', x, keepdims=False, axis=None)
            )
            assert_op_result(
                np.sum(x, axis=None, keepdims=True),
                wrap_op('reduce_sum', x, keepdims=True, axis=None)
            )

    def test_reduce_mean(self):
        with self.test_session():
            x = np.arange(24, dtype=np.float32).reshape([2, 2, 2, 3])
            assert_op_result(
                np.mean(x, axis=-1, keepdims=False),
                wrap_op('reduce_mean', x, axis=-1)
            )
            assert_op_result(
                np.mean(x, axis=(0, 2), keepdims=False),
                wrap_op('reduce_mean', x, keepdims=False, axis=(0, 2))
            )
            assert_op_result(
                np.mean(x, axis=(0, 2), keepdims=True),
                wrap_op('reduce_mean', x, keepdims=True, axis=(0, 2))
            )
            assert_op_result(
                np.mean(x, axis=None, keepdims=False),
                wrap_op('reduce_mean', x, keepdims=False, axis=None)
            )
            assert_op_result(
                np.mean(x, axis=None, keepdims=True),
                wrap_op('reduce_mean', x, keepdims=True, axis=None)
            )

    def test_reduce_max(self):
        with self.test_session():
            x = np.arange(24).reshape([2, 2, 2, 3])
            assert_op_result(
                np.max(x, axis=-1, keepdims=False),
                wrap_op('reduce_max', x, axis=-1)
            )
            assert_op_result(
                np.max(x, axis=(0, 2), keepdims=False),
                wrap_op('reduce_max', x, keepdims=False, axis=(0, 2))
            )
            assert_op_result(
                np.max(x, axis=(0, 2), keepdims=True),
                wrap_op('reduce_max', x, keepdims=True, axis=(0, 2))
            )
            assert_op_result(
                np.max(x, axis=None, keepdims=False),
                wrap_op('reduce_max', x, keepdims=False, axis=None)
            )
            assert_op_result(
                np.max(x, axis=None, keepdims=True),
                wrap_op('reduce_max', x, keepdims=True, axis=None)
            )

    def test_reduce_min(self):
        with self.test_session():
            x = np.arange(24).reshape([2, 2, 2, 3])
            assert_op_result(
                np.min(x, axis=-1, keepdims=False),
                wrap_op('reduce_min', x, axis=-1)
            )
            assert_op_result(
                np.min(x, axis=(0, 2), keepdims=False),
                wrap_op('reduce_min', x, keepdims=False, axis=(0, 2))
            )
            assert_op_result(
                np.min(x, axis=(0, 2), keepdims=True),
                wrap_op('reduce_min', x, keepdims=True, axis=(0, 2))
            )
            assert_op_result(
                np.min(x, axis=None, keepdims=False),
                wrap_op('reduce_min', x, keepdims=False, axis=None)
            )
            assert_op_result(
                np.min(x, axis=None, keepdims=True),
                wrap_op('reduce_min', x, keepdims=True, axis=None)
            )

    def test_squeeze(self):
        with self.test_session():
            x = np.asarray([2, 3]).reshape([1, 2, 1, 1])
            assert_op_result(
                np.squeeze(x),
                wrap_op('squeeze', x)
            )
            assert_op_result(
                np.squeeze(x, axis=-1),
                wrap_op('squeeze', x, axis=-1)
            )
            assert_op_result(
                np.squeeze(x, axis=(0, 2)),
                wrap_op('squeeze', x, axis=(0, 2))
            )
            assert_op_result(
                np.squeeze(x, axis=(0, 2, 3)),
                wrap_op('squeeze', x, axis=(0, 2, 3))
            )

    def test_range(self):
        with tf.Session().as_default():
            assert_op_result(
                np.arange(0),
                wrap_op('range', 0)
            )
            assert_op_result(
                np.arange(10),
                wrap_op('range', 10)
            )
            assert_op_result(
                np.arange(0, 10),
                wrap_op('range', 0, 10)
            )
            assert_op_result(
                np.arange(0, 10, 3),
                wrap_op('range', 0, 10, 3)
            )

    def test_shape_rank_reshape(self):
        with tf.Session().as_default():
            assert_op_result(
                [2, 2, 3, 2],
                wrap_op('shape', np.arange(24).reshape([2, 2, 3, 2]))
            )
            assert_op_result(
                4,
                wrap_op('rank', np.arange(24).reshape([2, 2, 3, 2]))
            )
            assert_op_result(
                np.arange(24).reshape([4, 6]),
                wrap_op('reshape', np.arange(24).reshape([2, 2, 3, 2]),
                        (4, 6))
            )

    def test_assert_rank_at_least(self):
        with tf.Session().as_default():
            for ops in [npyops, tfops]:
                x = np.arange(24).reshape([4, 6])
                with ops.assert_rank_at_least(x, 2):
                    assert_op_result(
                        x,
                        wrap_op('identity', x)
                    )
                with pytest.raises(Exception):
                    with ops.assert_rank_at_least(x, 3):
                        assert_op_result(
                            x,
                            wrap_op('identity', x)
                        )
                with pytest.raises(Exception, match='The Error Message'):
                    with ops.assert_rank_at_least(
                            x, 3, message='The Error Message'):
                        assert_op_result(
                            x,
                            wrap_op('identity', x)
                        )


if __name__ == '__main__':
    unittest.main()
