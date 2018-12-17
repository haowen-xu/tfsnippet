import unittest

import pytest
import tensorflow as tf

from tfsnippet.utils import *


def make_var_and_op(var_name, op_name):
    vs = tf.get_variable_scope()
    var = tf.get_variable(var_name, shape=(), dtype=tf.float32)
    op = tf.add(1, 2, name=op_name)
    return vs, var, op


class GetValidNameScopeNameTestCase(unittest.TestCase):

    def test_get_valid_name_scope_name(self):
        class _MyClass:
            pass

        self.assertEqual(get_valid_name_scope_name('abc'), 'abc')
        self.assertEqual(get_valid_name_scope_name('abc', str), 'str.abc')
        self.assertEqual(get_valid_name_scope_name('abc', ''), 'str.abc')
        self.assertEqual(get_valid_name_scope_name('abc', _MyClass),
                         'MyClass.abc')
        self.assertEqual(get_valid_name_scope_name('abc', _MyClass()),
                         'MyClass.abc')


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


class VarScopeObjectTestCase(tf.test.TestCase):

    def test_repr(self):
        class MyVarScopeObj(VarScopeObject):
            pass

        self.assertEqual(repr(MyVarScopeObj(name='a')), "MyVarScopeObj('a')")
        with tf.variable_scope('parent'):
            self.assertEqual(repr(MyVarScopeObj(scope='b')),
                             "MyVarScopeObj('parent/b')")

    def test_default_name(self):
        class MyVarScopeObj(VarScopeObject):
            pass

        self.assertEqual(MyVarScopeObj().variable_scope.name,
                         'my_var_scope_obj')
        self.assertEqual(MyVarScopeObj().variable_scope.name,
                         'my_var_scope_obj_1')

    def test_construct_with_name_and_scope(self):
        class MyVarScopeObj(VarScopeObject):
            pass

        # test the first object
        o1 = MyVarScopeObj(name='o')
        self.assertEqual(o1.name, 'o')
        self.assertEqual(o1.variable_scope.name, 'o')
        self.assertEqual(o1.variable_scope.original_name_scope, 'o/')

        # test the second object with the same default name
        o2 = MyVarScopeObj(name='o')
        self.assertEqual(o2.name, 'o')
        self.assertEqual(o2.variable_scope.name, 'o_1')
        self.assertEqual(o2.variable_scope.original_name_scope, 'o_1/')

        # test the third object with the same scope as o1
        o3 = MyVarScopeObj(scope='o')
        self.assertIsNone(o3.name)
        self.assertEqual(o3.variable_scope.name, 'o')
        self.assertEqual(o3.variable_scope.original_name_scope, 'o_2/')

        # test to construct object under other scope
        with tf.variable_scope('c'):
            o4 = MyVarScopeObj(name='o')
            self.assertEqual(o4.name, 'o')
            self.assertEqual(o4.variable_scope.name, 'c/o')
            self.assertEqual(o4.variable_scope.original_name_scope, 'c/o/')

    def test_override_variable_scope_created(self):
        class MyVarScopeObj(VarScopeObject):
            def _variable_scope_created(self, vs):
                self.vs = vs
                self.a = tf.get_variable('a', shape=(), dtype=tf.float32)

        o = MyVarScopeObj(name='o')
        self.assertEqual(o.vs.name, 'o')
        self.assertEqual(o.a.name, 'o/a:0')
