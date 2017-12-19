# -*- coding: utf-8 -*-
import numpy as np
import pytest
import six
import tensorflow as tf

from tfsnippet.utils import TensorWrapper, register_tensor_wrapper_class
from tests.utils._div_op import regular_div, floor_div
from tests.utils._true_div_op import true_div


class _SimpleTensor(TensorWrapper):

    def __init__(self, wrapped, flag=None):
        self._self_flag_ = flag
        self._self_tensor_ = wrapped
        super(_SimpleTensor, self).__init__()

    @property
    def tensor(self):
        return self._self_tensor_

    @property
    def flag(self):
        return self._self_flag_

    def get_flag(self):
        return self._self_flag_

register_tensor_wrapper_class(_SimpleTensor)


class TensorWrapperArithTestCase(tf.test.TestCase):

    def test_prerequisite(self):
        if six.PY2:
            self.assertAlmostEqual(regular_div(3, 2), 1)
            self.assertAlmostEqual(regular_div(3.3, 1.6), 2.0625)
        else:
            self.assertAlmostEqual(regular_div(3, 2), 1.5)
            self.assertAlmostEqual(regular_div(3.3, 1.6), 2.0625)
        self.assertAlmostEqual(true_div(3, 2), 1.5)
        self.assertAlmostEqual(true_div(3.3, 1.6), 2.0625)
        self.assertAlmostEqual(floor_div(3, 2), 1)
        self.assertAlmostEqual(floor_div(3.3, 1.6), 2.0)

    def test_unary_op(self):
        def check_op(name, func, x):
            x_tensor = tf.convert_to_tensor(x)
            ans = func(x_tensor)
            res = tf.convert_to_tensor(func(_SimpleTensor(x_tensor)))
            self.assertEqual(
                res.dtype, ans.dtype,
                msg='Result dtype does not match answer after unary operator '
                    '{} is applied: {!r} vs {!r} (x is {!r})'
                    .format(name, res.dtype, ans.dtype, x)
            )
            res_val = res.eval()
            ans_val = ans.eval()
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg='Result value does not match answer after unary '
                        'operator {} is applied: {!r} vs {!r} (x is {!r})'
                        .format(name, res_val, ans_val, x)
            )

        with self.test_session():
            int_data = np.asarray([1, -2, 3], dtype=np.int32)
            float_data = np.asarray([1.1, -2.2, 3.3], dtype=np.float32)
            bool_data = np.asarray([True, False, True], dtype=np.bool)

            check_op('abs', abs, int_data)
            check_op('abs', abs, float_data)
            check_op('neg', (lambda v: -v), int_data)
            check_op('neg', (lambda v: -v), float_data)
            check_op('invert', (lambda v: ~v), bool_data)

    def test_binary_op(self):
        def check_op(name, func, x, y):
            x_tensor = tf.convert_to_tensor(x)
            y_tensor = tf.convert_to_tensor(y)
            ans = func(x_tensor, y_tensor)
            res_1 = tf.convert_to_tensor(
                func(_SimpleTensor(x_tensor), y))
            res_2 = tf.convert_to_tensor(
                func(x, _SimpleTensor(y_tensor)))
            res_3 = tf.convert_to_tensor(
                func(_SimpleTensor(x_tensor), y_tensor))
            res_4 = tf.convert_to_tensor(
                func(x_tensor, _SimpleTensor(y_tensor)))
            res_5 = tf.convert_to_tensor(
                func(_SimpleTensor(x_tensor), _SimpleTensor(y_tensor)))

            for tag, res in [('TensorWrapper + np.ndarray', res_1),
                             ('np.ndarray + TensorWrapper', res_2),
                             ('TensorWrapper + Tensor', res_3),
                             ('Tensor + TensorWrapper', res_4),
                             ('TensorWrapper + TensorWrapper', res_5)]:
                self.assertEqual(
                    res.dtype, ans.dtype,
                    msg='Result dtype does not match answer after {} binary '
                        'operator {} is applied: {!r} vs {!r} (x is {!r}, '
                        'y is {!r})'.
                        format(tag, name, res.dtype, ans.dtype, x, y)
                )
                res_val = res.eval()
                ans_val = ans.eval()
                np.testing.assert_equal(
                    res_val, ans_val,
                    err_msg='Result value does not match answer after {} '
                            'binary operator {} is applied: {!r} vs {!r} '
                            '(x is {!r}, y is {!r}).'
                            .format(tag, name, res_val, ans_val, x, y)
                )

        def run_ops(x, y, ops):
            for name, func in six.iteritems(ops):
                check_op(name, func, x, y)

        arith_ops = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': regular_div,
            'truediv': true_div,
            'floordiv': floor_div,
            'mod': lambda x, y: x % y,
        }

        logical_ops = {
            'and': lambda x, y: x & y,
            'or': lambda x, y: x | y,
            'xor': lambda x, y: x ^ y,
        }

        relation_ops = {
            'lt': lambda x, y: x < y,
            'le': lambda x, y: x <= y,
            'gt': lambda x, y: x > y,
            'ge': lambda x, y: x >= y,
        }

        with self.test_session():
            # arithmetic operators
            run_ops(np.asarray([-4, 5, 6], dtype=np.int32),
                    np.asarray([1, -2, 3], dtype=np.int32),
                    arith_ops)
            run_ops(np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                    np.asarray([1.1, -2.2, 3.3], dtype=np.float32),
                    arith_ops)

            # it seems that tf.pow(x, y) does not support negative integers
            # yet, so we individually test this operator here.
            check_op('pow',
                     (lambda x, y: x ** y),
                     np.asarray([-4, 5, 6], dtype=np.int32),
                     np.asarray([1, 2, 3], dtype=np.int32))
            check_op('pow',
                     (lambda x, y: x ** y),
                     np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                     np.asarray([1.1, -2.2, 3.3], dtype=np.float32))

            # logical operators
            run_ops(np.asarray([True, False, True, False], dtype=np.bool),
                    np.asarray([True, True, False, False], dtype=np.bool),
                    logical_ops)

            # relation operators
            run_ops(np.asarray([1, -2, 3, -4, 5, 6, -4, 5, 6], dtype=np.int32),
                    np.asarray([1, -2, 3, 1, -2, 3, -4, 5, 6], dtype=np.int32),
                    relation_ops)
            run_ops(
                np.asarray([1.1, -2.2, 3.3, -4.4, 5.5, 6.6, -4.4, 5.5, 6.6],
                           dtype=np.float32),
                np.asarray([1.1, -2.2, 3.3, 1.1, -2.2, 3.3, -4.4, 5.5, 6.6],
                           dtype=np.float32),
                relation_ops
            )

    def test_getitem(self):
        def check_getitem(x, y, xx, yy):
            ans = tf.convert_to_tensor(x[y])
            res = xx[yy]

            self.assertEqual(
                res.dtype, ans.dtype,
                msg='Result dtype does not match answer after getitem '
                    'is applied: {!r} vs {!r} (x is {!r}, y is {!r}, '
                    'xx is {!r}, yy is {!r}).'.
                    format(res.dtype, ans.dtype, x, y, xx, yy)
            )
            res_val = res.eval()
            ans_val = ans.eval()
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg='Result value does not match answer after '
                        'getitem is applied: {!r} vs {!r} (x is {!r}, '
                        'y is {!r}, xx is {!r}, yy is {!r}).'
                        .format(res_val, ans_val, x, y, xx, yy)
            )

        class _SliceGenerator(object):
            def __getitem__(self, item):
                return item
        sg = _SliceGenerator()

        with self.test_session():
            data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
            indices_or_slices = [
                0,
                -1,
                # TensorFlow has not supported array index yet.
                # np.asarray([0, 3, 2, 6], dtype=np.int32),
                # np.asarray([-1, -2, -3], dtype=np.int32),
                sg[0:],
                sg[:1],
                sg[:: 2],
                sg[-1:],
                sg[: -1],
                sg[:: -1],
            ]
            for s in indices_or_slices:
                x_tensor = tf.convert_to_tensor(data)
                x_simple_tensor = _SimpleTensor(x_tensor)
                check_getitem(data, s, x_simple_tensor, s)

                if not isinstance(s, slice):
                    y_tensor = tf.convert_to_tensor(s)
                    y_simple_tensor = _SimpleTensor(y_tensor)
                    check_getitem(data, s, x_simple_tensor, y_tensor)
                    check_getitem(data, s, x_simple_tensor, y_simple_tensor)
                    check_getitem(data, s, x_tensor, y_simple_tensor)


class TensorWrapperInterfaceTestCase(tf.test.TestCase):

    def test_disallowed_op(self):
        with pytest.raises(
                TypeError, match='`_SimpleTensor` is not iterable'):
            _ = iter(_SimpleTensor(tf.constant(1)))

        with pytest.raises(
                TypeError, match='Using a `_SimpleTensor` as a Python `bool` '
                                 'is not allowed.*'):
            _ = not _SimpleTensor(tf.constant(1))

        with pytest.raises(
                TypeError, match='Using a `_SimpleTensor` as a Python `bool` '
                                 'is not allowed.*'):
            if _SimpleTensor(tf.constant(1)):
                pass

    def test_convert_to_tensor(self):
        with self.test_session():
            t = _SimpleTensor(tf.constant(1.))
            self.assertIsInstance(tf.convert_to_tensor(t), tf.Tensor)
            self.assertNotIsInstance(tf.convert_to_tensor(t), _SimpleTensor)

    def test_error_convert_to_tensor(self):
        with pytest.raises(
                ValueError, match='Incompatible type conversion requested to '
                                  'type int32 for tensor of type float32'):
            _ = tf.convert_to_tensor(
                _SimpleTensor(tf.constant(1., dtype=tf.float32)),
                dtype=tf.int32
            )

    def test_session_run(self):
        with self.test_session() as sess:
            # test session run
            t = _SimpleTensor(tf.constant([1., 2., 3.]))
            np.testing.assert_equal(sess.run(t), [1., 2., 3.])

            # test using in feed_dict
            np.testing.assert_equal(
                sess.run(tf.identity(t), feed_dict={
                    t: np.asarray([4., 5., 6.])
                }),
                np.asarray([4., 5., 6.])
            )

    def test_get_attributes(self):
        t = _SimpleTensor(tf.constant([1., 2., 3.]), flag=123)
        self.assertEqual(t.flag, 123)
        self.assertEqual(t._self_flag_, 123)
        self.assertEqual(t.get_flag(), 123)
        members = dir(t)
        for member in ['flag', '_self_flag_', 'get_flag',
                       '_self_tensor_', 'tensor']:
            self.assertIn(
                member, members,
                msg='{!r} should in dir(t), but not'.format(members)
            )
            self.assertTrue(
                hasattr(t, member),
                msg='_SimpleTensor should has member {!r}, but not.'.
                    format(member)
            )
        for member in dir(t.tensor):
            if not member.startswith('_'):
                self.assertIn(
                    member, members,
                    msg='{!r} should in dir(t), but not'.format(members)
                )
                self.assertTrue(
                    hasattr(t, member),
                    msg='_SimpleTensor should has member {!r}, but not.'.
                        format(member,)
                )
                self.assertEqual(getattr(t, member),
                                 getattr(t.tensor, member))

    def test_set_attributes(self):
        t = _SimpleTensor(tf.constant([1., 2., 3.]))

        self.assertTrue(hasattr(t, '_self_flag_'))
        self.assertFalse(hasattr(t.tensor, '_self_flag_'))
        t._self_flag_ = 123
        self.assertEqual(t._self_flag_, 123)
        self.assertFalse(hasattr(t.tensor, '_self_flag_'))

        self.assertTrue(hasattr(t, 'get_flag'))
        self.assertFalse(hasattr(t.tensor, 'get_flag'))
        t.get_flag = 456
        self.assertEqual(t.get_flag, 456)
        self.assertTrue(hasattr(t, 'get_flag'))
        self.assertFalse(hasattr(t.tensor, 'get_flag'))

        self.assertTrue(hasattr(t, 'get_shape'))
        self.assertTrue(hasattr(t.tensor, 'get_shape'))
        t.get_shape = 789
        self.assertEqual(t.get_shape, 789)
        self.assertEqual(t.tensor.get_shape, 789)
        self.assertTrue(hasattr(t, 'get_shape'))
        self.assertTrue(hasattr(t.tensor, 'get_shape'))

        t.abc = 1001
        self.assertEqual(t.abc, 1001)
        self.assertEqual(t.tensor.abc, 1001)
        self.assertTrue(hasattr(t, 'abc'))
        self.assertTrue(hasattr(t.tensor, 'abc'))

        t.tensor.xyz = 2002
        self.assertEqual(t.xyz, 2002)
        self.assertEqual(t.tensor.xyz, 2002)
        self.assertTrue(hasattr(t, 'xyz'))
        self.assertTrue(hasattr(t.tensor, 'xyz'))

    def test_del_attributes(self):
        t = _SimpleTensor(tf.constant([1., 2., 3.]), flag=123)

        del t._self_flag_
        self.assertFalse(hasattr(t, '_self_flag_'))
        self.assertFalse(hasattr(t.tensor, '_self_flag_'))

        t.abc = 1001
        del t.abc
        self.assertFalse(hasattr(t, 'abc'))
        self.assertFalse(hasattr(t.tensor, 'abc'))

        t.tensor.xyz = 2002
        del t.xyz
        self.assertFalse(hasattr(t, 'xyz'))
        self.assertFalse(hasattr(t.tensor, 'xyz'))

        t.get_flag = 123
        del t.get_flag
        self.assertFalse(hasattr(t.tensor, 'get_flag'))
        self.assertNotEqual(t.get_flag, 123)


class _NonTensorWrapperClass(object):
    pass


class RegisterTensorWrapperClassTestCase(tf.test.TestCase):

    def test_register_non_tensor_wrapper_class(self):
        with pytest.raises(
                TypeError, match='`.*_NonTensorWrapperClass.*` is not a type, '
                                 'or not a subclass of `TensorWrapper`'):
            register_tensor_wrapper_class(_NonTensorWrapperClass)
        with pytest.raises(
                TypeError, match='`123` is not a type, or not a subclass of '
                                 '`TensorWrapper`'):
            register_tensor_wrapper_class(123)
