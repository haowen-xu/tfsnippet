import unittest

import pytest
import six
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.stochastic import StochasticTensor
from tfsnippet.utils import (is_integer, is_float, TensorWrapper,
                             is_tensor_object, TensorArgValidator)

if six.PY2:
    LONG_MAX = long(1) << 63 - long(1)
else:
    LONG_MAX = 1 << 63 - 1


class IsIntegerTestCase(unittest.TestCase):

    def test_is_integer(self):
        if six.PY2:
            self.assertTrue(is_integer(long(1)))
        self.assertTrue(is_integer(int(1)))
        for dtype in [np.int, np.int8, np.int16, np.int32, np.int64,
                      np.uint, np.uint8, np.uint16, np.uint32, np.uint64]:
            v = np.asarray([1], dtype=dtype)[0]
            self.assertTrue(
                is_integer(v),
                msg='{!r} should be interpreted as integer'.format(v)
            )
        self.assertFalse(is_integer(np.asarray(0, dtype=np.int)))
        for v in [float(1.0), '', object(), None, True, (), {}, []]:
            self.assertFalse(
                is_integer(v),
                msg='{!r} should not be interpreted as integer'.format(v)
            )


class IsFloatTestCase(unittest.TestCase):

    def test_is_float(self):
        float_types = [float, np.float, np.float16, np.float32, np.float64]
        for extra_type in ['float8', 'float128', 'float256']:
            if hasattr(np, extra_type):
                float_types.append(getattr(np, extra_type))
        for dtype in float_types:
            v = np.asarray([1], dtype=dtype)[0]
            self.assertTrue(
                is_float(v),
                msg='{!r} should be interpreted as float'.format(v)
            )
        self.assertFalse(is_integer(np.asarray(0., dtype=np.float32)))
        for v in [int(1), '', object(), None, True, (), {}, []]:
            self.assertFalse(
                is_float(v),
                msg='{!r} should not be interpreted as float'.format(v)
            )


class IsTensorObjectTestCase(unittest.TestCase):

    def test_is_tensor_object(self):
        for obj in [tf.constant(0.),  # type: tf.Tensor
                    tf.get_variable('x', dtype=tf.float32, shape=()),
                    TensorWrapper(),
                    StochasticTensor(Mock(is_reparameterized=False),
                                     tf.constant(0.))]:
            self.assertTrue(
                is_tensor_object(obj),
                msg='{!r} should be interpreted as a tensor object'.format(obj)
            )

        for obj in [1, '', object(), None, True, (), {}, [], np.zeros([1])]:
            self.assertFalse(
                is_tensor_object(obj),
                msg='{!r} should not be interpreted as a tensor object'.
                    format(obj)
            )


class TensorArgValidatorTestCase(tf.test.TestCase):

    def test_require_int32(self):
        v = TensorArgValidator('xyz')

        # test static values
        for o in [0, 1, -1]:
            self.assertEqual(v.require_int32(o), o)

        for o in [object(), None, (), [], 1.2, LONG_MAX]:
            with pytest.raises(TypeError,
                               match='xyz cannot be converted to int32'):
                _ = v.require_int32(o)

        # test dynamic values
        with self.test_session():
            for o in [0, 1, -1]:
                self.assertEqual(
                    v.require_int32(tf.constant(o, dtype=tf.int32)).eval(), o)

            for o in [tf.constant(1.2, dtype=tf.float32),
                      tf.constant(LONG_MAX, dtype=tf.int64)]:
                with pytest.raises(TypeError,
                                   match='xyz cannot be converted to int32'):
                    _ = v.require_int32(o)

    def test_require_non_negative(self):
        v = TensorArgValidator('xyz')

        # test static values
        for o in [0, 0., 1e-7, 1, 1.]:
            self.assertEqual(v.require_non_negative(o), o)

        for o in [-1., -1, -1e-7]:
            with pytest.raises(ValueError, match='xyz must be non-negative'):
                _ = v.require_non_negative(o)

        # test dynamic values
        with self.test_session():
            for o, dtype in zip(
                    [0, 0., 1e-7, 1, 1.],
                    [tf.int32, tf.float32, tf.float32, tf.int32, tf.float32]):
                self.assertAllClose(
                    v.require_non_negative(tf.constant(o, dtype=dtype)).eval(),
                    o
                )

            for o, dtype in zip(
                    [-1., -1, -1e-7], [tf.float32, tf.int32, tf.float32]):
                with pytest.raises(Exception, match='xyz must be non-negative'):
                    _ = v.require_non_negative(
                        tf.constant(o, dtype=dtype)).eval()

    def test_require_positive(self):
        v = TensorArgValidator('xyz')

        # test static values
        for o in [1e-7, 1, 1.]:
            self.assertEqual(v.require_positive(o), o)

        for o in [-1., -1, -1e-7, 0., 0]:
            with pytest.raises(ValueError, match='xyz must be positive'):
                _ = v.require_positive(o)

        # test dynamic values
        with self.test_session():
            for o, dtype in zip(
                    [1e-7, 1, 1.], [tf.float32, tf.int32, tf.float32]):
                self.assertAllClose(
                    v.require_positive(tf.constant(o, dtype=dtype)).eval(), o)

            for o, dtype in zip(
                    [-1., -1, -1e-7, 0., 0],
                    [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32]):
                with pytest.raises(Exception, match='xyz must be positive'):
                    _ = v.require_positive(tf.constant(o, dtype=dtype)).eval()
