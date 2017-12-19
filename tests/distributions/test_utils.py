import six
import pytest
import numpy as np
import tensorflow as tf

from tfsnippet.distributions import validate_group_ndims, reduce_group_ndims

if six.PY2:
    LONG_MAX = long(1) << 63 - long(1)
else:
    LONG_MAX = 1 << 63 - 1


class ValidateGroupNdimsTestCase(tf.test.TestCase):

    def test_static_values(self):
        # type checks
        for o in [object(), None, 1.2, LONG_MAX]:
            with pytest.raises(
                    TypeError,
                    match='group_ndims cannot be converted to int32'):
                _ = validate_group_ndims(o)

        # value checks
        self.assertEqual(validate_group_ndims(0), 0)
        self.assertEqual(validate_group_ndims(1), 1)
        with pytest.raises(
                ValueError, match='group_ndims must be non-negative'):
            _ = validate_group_ndims(-1)

    def test_dynamic_values(self):
        # type checks
        for o in [tf.constant(1.2, dtype=tf.float32),
                  tf.constant(LONG_MAX, dtype=tf.int64)]:
            with pytest.raises(
                    TypeError,
                    match='group_ndims cannot be converted to int32'):
                _ = validate_group_ndims(o)

        # value checks
        with self.test_session():
            self.assertEqual(
                validate_group_ndims(tf.constant(0, dtype=tf.int32)).eval(), 0)
            self.assertEqual(
                validate_group_ndims(tf.constant(1, dtype=tf.int32)).eval(), 1)
            with pytest.raises(
                    Exception, match='group_ndims must be non-negative'):
                _ = validate_group_ndims(tf.constant(-1, dtype=tf.int32)).eval()


class ReduceGroupNdimsTestCase(tf.test.TestCase):

    def test_errors(self):
        for o in [object(), None, 1.2, LONG_MAX,
                  tf.constant(1.2, dtype=tf.float32),
                  tf.constant(LONG_MAX, dtype=tf.int64)]:
            with pytest.raises(
                    TypeError,
                    match='group_ndims cannot be converted to int32'):
                _ = reduce_group_ndims(tf.reduce_sum, tf.constant(0.), o)

        with pytest.raises(
                ValueError, match='group_ndims must be non-negative'):
            _ = reduce_group_ndims(tf.reduce_sum, tf.constant(0.), -1)

        with self.test_session():
            with pytest.raises(
                    Exception, match='group_ndims must be non-negative'):
                _ = reduce_group_ndims(tf.reduce_sum, tf.constant(0.),
                                       tf.constant(-1, dtype=tf.int32)).eval()

    def test_output(self):
        tensor = tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4])
        tensor_sum_1 = tf.reduce_sum(tensor, axis=-1)
        tensor_sum_2 = tf.reduce_sum(tensor, axis=[-2, -1])
        tensor_prod = tf.reduce_prod(tensor, axis=-1)
        g0 = tf.constant(0, dtype=tf.int32)
        g1 = tf.constant(1, dtype=tf.int32)
        g2 = tf.constant(2, dtype=tf.int32)

        with self.test_session():
            # static group_ndims
            np.testing.assert_equal(
                tensor.eval(),
                reduce_group_ndims(tf.reduce_sum, tensor, 0).eval()
            )
            np.testing.assert_equal(
                tensor_sum_1.eval(),
                reduce_group_ndims(tf.reduce_sum, tensor, 1).eval()
            )
            np.testing.assert_equal(
                tensor_sum_2.eval(),
                reduce_group_ndims(tf.reduce_sum, tensor, 2).eval()
            )
            np.testing.assert_equal(
                tensor_prod.eval(),
                reduce_group_ndims(tf.reduce_prod, tensor, 1).eval()
            )

            # dynamic group_ndims
            np.testing.assert_equal(
                tensor.eval(),
                reduce_group_ndims(tf.reduce_sum, tensor, g0).eval()
            )
            np.testing.assert_equal(
                tensor_sum_1.eval(),
                reduce_group_ndims(tf.reduce_sum, tensor, g1).eval()
            )
            np.testing.assert_equal(
                tensor_sum_2.eval(),
                reduce_group_ndims(tf.reduce_sum, tensor, g2).eval()
            )
            np.testing.assert_equal(
                tensor_prod.eval(),
                reduce_group_ndims(tf.reduce_prod, tensor, g1).eval()
            )
