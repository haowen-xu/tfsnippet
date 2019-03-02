import functools
import unittest

import pytest
import tensorflow as tf

from tfsnippet.utils import *
from tfsnippet.utils.tensor_spec import TensorSpec


class TensorSpecTestCase(tf.test.TestCase):

    def test_repr_and_properties(self):
        # test empty spec
        s = TensorSpec()
        self.assertEqual(repr(s), 'TensorSpec()')
        self.assertIsNone(s.shape)
        self.assertIsNone(s.value_shape)
        self.assertIsNone(s.value_ndims)
        self.assertIsNone(s.dtype)

        # test empty spec by shape == ('...',)
        s = TensorSpec(shape=('...',))
        self.assertEqual(repr(s), 'TensorSpec()')
        self.assertIsNone(s.shape)
        self.assertIsNone(s.value_shape)
        self.assertIsNone(s.value_ndims)
        self.assertIsNone(s.dtype)

        # test shape spec
        s = TensorSpec(shape=('...', '?', '*', '2?', 3, -1, None))
        self.assertEqual(repr(s), 'TensorSpec(shape=(...,?,*,2?,3,?,?))')
        self.assertEqual(s.shape, ('...', '?', '*', '2?', 3, '?', '?'))
        self.assertEqual(s.value_shape, ('?', '*', '2?', 3, '?', '?'))
        self.assertEqual(s.value_ndims, 6)

        # test one element shape spec
        s = TensorSpec(shape=(1,))
        self.assertEqual(repr(s), 'TensorSpec(shape=(1,))')
        self.assertEqual(s.shape, (1,))
        self.assertEqual(s.value_shape, (1,))
        self.assertEqual(s.value_ndims, 1)

        # test dtype spec
        s = TensorSpec(dtype=tf.int64)
        self.assertEqual(repr(s), 'TensorSpec(dtype=int64)')
        self.assertEqual(s.dtype, tf.int64)

        # test shape & dtype spec
        s = TensorSpec(shape=(), dtype=tf.int32)
        self.assertEqual(repr(s), 'TensorSpec(shape=(),dtype=int32)')
        self.assertEqual(s.shape, ())
        self.assertEqual(s.value_shape, ())
        self.assertEqual(s.value_ndims, 0)
        self.assertEqual(s.dtype, tf.int32)

        # test derived class
        s = InputSpec(shape=('...', 1), dtype=tf.int64)
        self.assertEqual(repr(s), 'InputSpec(shape=(...,1),dtype=int64)')
        self.assertEqual(s.shape, ('...', 1))
        self.assertEqual(s.value_shape, (1,))
        self.assertEqual(s.value_ndims, 1)

    def test_equal_and_hash(self):
        equivalent_classes = [
            [TensorSpec(), TensorSpec()],
            [TensorSpec(dtype=tf.int32), TensorSpec(dtype=tf.int32)],
            [TensorSpec(dtype=tf.int64), TensorSpec(dtype=tf.int64)],
            [TensorSpec(shape=(1, 2)), TensorSpec(shape=(1, 2))],
            [TensorSpec(shape=(1, 3), dtype=tf.int32),
             TensorSpec(shape=(1, 3), dtype=tf.int32)],
            [TensorSpec(shape=(1, 3), dtype=tf.int64),
             TensorSpec(shape=(1, 3), dtype=tf.int64)],
            [TensorSpec(shape=(1, 2), dtype=tf.int64),
             TensorSpec(shape=(1, 2), dtype=tf.int64)],
            [TensorSpec(shape=('...', 1, 2), dtype=tf.int64),
             TensorSpec(shape=('...', 1, 2), dtype=tf.int64)],
        ]

        for equivalent_class in equivalent_classes:
            a, b = equivalent_class
            self.assertEqual(a, b)
            self.assertEqual(hash(a), hash(b))

        for cls1, cls2 in zip(equivalent_classes[1:], equivalent_classes[:-1]):
            self.assertNotEqual(cls1[0], cls2[0])

    def test_parse_error(self):
        with pytest.raises(ValueError, match='`...` should only be the first '
                                             'item of `shape`'):
            _ = TensorSpec((1, '...'))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec(('x',))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec((object(),))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec((0,))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec((-2,))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec(('0?',))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec(('-1?',))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec(('-2?',))
        with pytest.raises(ValueError, match='Invalid value in `shape`'):
            _ = TensorSpec(('x?',))

    def test_validate(self):
        def good(s, t):
            t = tf.convert_to_tensor(t)
            self.assertIs(s.validate('x', t), t)

        def bad(s, t, err_class, err_msg):
            t = tf.convert_to_tensor(t)
            with pytest.raises(err_class, match=err_msg):
                _ = s.validate('x', t)

        bad_dtype = functools.partial(
            bad, err_class=TypeError, err_msg='The dtype of `x` is invalid')
        bad_shape = functools.partial(
            bad, err_class=ValueError, err_msg='The shape of `x` is invalid')

        # test empty validation
        s = TensorSpec()
        good(s, tf.constant(1))
        good(s, tf.placeholder(tf.float32, None))

        # test dtype
        s = TensorSpec(dtype=tf.int64)
        good(s, tf.constant(1, tf.int64))
        bad_dtype(s, tf.constant(1., tf.float32))

        # test empty shape
        s = TensorSpec(shape=())
        good(s, tf.placeholder(tf.float32, ()))
        bad_shape(s, tf.placeholder(tf.float32, None))
        bad_shape(s, tf.placeholder(tf.float32, [2]))
        bad_shape(s, tf.placeholder(tf.float32, [2, 2]))
        bad_shape(s, tf.placeholder(tf.float32, [3, 1, 2]))

        # test static shape
        s = TensorSpec(shape=(1, 2))
        good(s, tf.placeholder(tf.float32, [1, 2]))
        bad_shape(s, tf.placeholder(tf.float32, None))
        bad_shape(s, tf.placeholder(tf.float32, ()))
        bad_shape(s, tf.placeholder(tf.float32, [2]))
        bad_shape(s, tf.placeholder(tf.float32, [2, 2]))
        bad_shape(s, tf.placeholder(tf.float32, [3, 1, 2]))

        # test dynamic shape
        s = TensorSpec(shape=('?', 3))
        good(s, tf.placeholder(tf.float32, [2, 3]))
        good(s, tf.placeholder(tf.float32, [None, 3]))
        bad_shape(s, tf.placeholder(tf.float32, None))
        bad_shape(s, tf.placeholder(tf.float32, ()))
        bad_shape(s, tf.placeholder(tf.float32, [2]))
        bad_shape(s, tf.placeholder(tf.float32, [2, 1]))
        bad_shape(s, tf.placeholder(tf.float32, [3, 2, 3]))

        # test allow more dims
        s = TensorSpec(shape=('...', '?', 3))
        good(s, tf.placeholder(tf.float32, [2, 3]))
        good(s, tf.placeholder(tf.float32, [None, 3]))
        good(s, tf.placeholder(tf.float32, [3, 2, 3]))
        good(s, tf.placeholder(tf.float32, [None, None, 3]))
        bad_shape(s, tf.placeholder(tf.float32, None))
        bad_shape(s, tf.placeholder(tf.float32, ()))
        bad_shape(s, tf.placeholder(tf.float32, [2]))
        bad_shape(s, tf.placeholder(tf.float32, [2, 1]))
        bad_shape(s, tf.placeholder(tf.float32, [3, 2, 1]))

        # test unknown but fixed dimensions
        s = TensorSpec(shape=('*', 3))
        good(s, tf.placeholder(tf.float32, [2, 3]))
        bad_shape(s, tf.placeholder(tf.float32, [None, 3]))
        bad_shape(s, tf.placeholder(tf.float32, [3]))

        # test known or dynamic dimensions
        s = TensorSpec(shape=('2?', 3))
        good(s, tf.placeholder(tf.float32, [2, 3]))
        good(s, tf.placeholder(tf.float32, [None, 3]))
        bad_shape(s, tf.placeholder(tf.float32, [4, 3]))
        bad_shape(s, tf.placeholder(tf.float32, [3]))


class ParamSpecTestCase(unittest.TestCase):

    def test_invalid_shape(self):
        def good(shape):
            _ = ParamSpec(shape=shape)

        good(())
        good((1, 2))

        def bad(shape, shape_msg):
            shape_msg = shape_msg.replace('(', '\\(')
            shape_msg = shape_msg.replace(')', '\\)')
            shape_msg = shape_msg.replace('?', '\\?')
            shape_msg = shape_msg.replace('*', '\\*')
            with pytest.raises(ValueError, match='The shape of a `ParamSpec` '
                                                 'must be fully determined: '
                                                 'got {}'.format(shape_msg)):
                _ = ParamSpec(shape=shape)

        bad(None, 'None')
        bad(('...',), 'None')
        bad(('...', 1), '(...,1)')
        bad(('1?',), '(1?,)')
        bad(('*',), '(*,)')
        bad(('?',), '(?,)')
