import unittest

import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.utils import *


def make_var_and_op(var_name, op_name):
    vs = tf.get_variable_scope()
    var = tf.get_variable(var_name, shape=(), dtype=tf.float32)
    op = tf.add(1, 2, name=op_name)
    return vs, var, op


class GetDefaultScopeNameTestCase(unittest.TestCase):

    def test_get_default_scope_name(self):
        class _MyClass:
            pass

        class _MyClass2:
            variable_scope = Mock(tf.VariableScope)
        _MyClass2.variable_scope.name = 'x'

        self.assertEqual(get_default_scope_name('abc'), 'abc')
        self.assertEqual(get_default_scope_name('abc', str), 'str.abc')
        self.assertEqual(get_default_scope_name('abc', ''), 'str.abc')
        self.assertEqual(get_default_scope_name('abc', _MyClass),
                         'MyClass.abc')
        self.assertEqual(get_default_scope_name('abc', _MyClass()),
                         'MyClass.abc')
        self.assertEqual(get_default_scope_name('abc', _MyClass2), 'x.abc')
        self.assertEqual(get_default_scope_name('abc', _MyClass2()), 'x.abc')


class ReopenVariableScopeTestCase(tf.test.TestCase):

    def test_reopen_root_variable_scope(self):
        with tf.Graph().as_default():
            root = tf.get_variable_scope()

            # test to reopen root within root
            with reopen_variable_scope(root):
                vs, v1, op = make_var_and_op('v1', 'op')
                self.assertEqual(vs.name, '')
                self.assertEqual(v1.name, 'v1:0')
                self.assertEqual(op.name, 'op:0')

            # test to reopen root within another variable scope
            with tf.variable_scope('a') as a:
                with reopen_variable_scope(root):
                    vs, v2, op = make_var_and_op('v2', 'op')
                    self.assertEqual(vs.name, '')
                    self.assertEqual(v2.name, 'v2:0')
                    self.assertEqual(op.name, 'op_1:0')

    def test_reopen_variable_scope(self):
        with tf.Graph().as_default():
            with tf.variable_scope('the_scope') as the_scope:
                pass

            # test to reopen within root
            with reopen_variable_scope(the_scope):
                vs, v1, op = make_var_and_op('v1', 'op')
                self.assertEqual(vs.name, 'the_scope')
                self.assertEqual(v1.name, 'the_scope/v1:0')
                self.assertEqual(op.name, 'the_scope/op:0')

            # test to reopen within another variable scope
            with tf.variable_scope('another'):
                with reopen_variable_scope(the_scope):
                    vs, v2, op = make_var_and_op('v2', 'op')
                    self.assertEqual(vs.name, 'the_scope')
                    self.assertEqual(v2.name, 'the_scope/v2:0')
                    self.assertEqual(op.name, 'the_scope/op_1:0')

    def test_errors(self):
        with pytest.raises(TypeError, match='`var_scope` must be an instance '
                                            'of `tf.VariableScope`'):
            with reopen_variable_scope(object()):
                pass


class RootVariableScopeTestCase(tf.test.TestCase):

    def test_root_variable_scope(self):
        with tf.Graph().as_default():
            # open root vs within the root vs
            with root_variable_scope() as root:
                vs, v1, op = make_var_and_op('v1', 'op')
                self.assertIs(vs, root)
                self.assertEqual(vs.name, '')
                self.assertEqual(v1.name, 'v1:0')
                self.assertEqual(op.name, 'op:0')

                # enter a new variable scope with determined name
                with tf.variable_scope('vs'):
                    vs, v2, op = make_var_and_op('v2', 'op')
                    self.assertEqual(vs.name, 'vs')
                    self.assertEqual(v2.name, 'vs/v2:0')
                    self.assertEqual(op.name, 'vs/op:0')

                # enter a reused variable scope with determined name
                with tf.variable_scope('vs', reuse=True):
                    vs, v2, op = make_var_and_op('v2', 'op')
                    self.assertEqual(vs.name, 'vs')
                    self.assertEqual(v2.name, 'vs/v2:0')
                    self.assertEqual(op.name, 'vs_1/op:0')

                # enter a new variable scope with uniquified name
                with tf.variable_scope(None, default_name='vs'):
                    vs, v3, op = make_var_and_op('v3', 'op')
                    self.assertEqual(vs.name, 'vs_1')
                    self.assertEqual(v3.name, 'vs_1/v3:0')
                    # THE FOLLOWING FACT IS ASTOUNDING BUT IS TRUE.
                    self.assertEqual(op.name, 'vs_2/op:0')

            # open the root vs within a parent scope
            # using root_variable_scope will cause the name scope to be
            # reset to the root name scope.
            with tf.variable_scope('parent'):
                with root_variable_scope():
                    vs, v4, op = make_var_and_op('v4', 'op')
                    self.assertEqual(vs.name, '')
                    self.assertEqual(v4.name, 'v4:0')
                    self.assertEqual(op.name, 'op_1:0')

            # as a contrary, if we open the root scope directly using
            # tf.variable_scope, we will end up with still the nested
            # name scope
            with tf.variable_scope('outside'):
                with tf.variable_scope(root):
                    vs, v5, op = make_var_and_op('v5', 'op')
                    self.assertEqual(vs.name, '')
                    self.assertEqual(v5.name, 'v5:0')
                    self.assertEqual(op.name, 'outside/op:0')
