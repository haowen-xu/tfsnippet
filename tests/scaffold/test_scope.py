import pytest
import tensorflow as tf

from tfsnippet.utils import is_tensorflow_version_higher_or_equal
from tfsnippet.scaffold import (reopen_variable_scope,
                                root_variable_scope,
                                VarScopeObject,
                                instance_reuse)


class _VarScopeTestMixin:

    def _check_ns(self, ns, ns_name, op_name):
        self.assertEqual(ns, ns_name)
        self.assertEqual(tf.add(1, 2, name='op').name, op_name)

    def _check_vs(self, get_var_name, vs_name, vs_scope, var_name, op_name):
        vs = tf.get_variable_scope()
        self.assertEqual(vs.name, vs_name)
        self.assertEqual(vs.original_name_scope, vs_scope)
        self.assertEqual(tf.get_variable(get_var_name, shape=()).name, var_name)
        self.assertEqual(tf.add(1, 2, name='op').name, op_name)


class TensorFlowScopeTestCase(tf.test.TestCase, _VarScopeTestMixin):

    def test_name_scope(self):
        with tf.name_scope(None) as root:
            self._check_ns(root, '', 'op:0')

            with tf.name_scope('a') as a:
                self._check_ns(a, 'a/', 'a/op:0')
                with tf.name_scope('') as ns:
                    self._check_ns(ns, '', 'op_1:0')
                    with tf.name_scope('a') as ns:
                        self._check_ns(ns, 'a_1/', 'a_1/op:0')

            with tf.name_scope('b/c') as ns:
                self._check_ns(ns, 'b/c/', 'b/c/op:0')

            with tf.name_scope('b') as ns:
                self._check_ns(ns, 'b/', 'b/op:0')
                with tf.name_scope('c') as ns:
                    self._check_ns(ns, 'b/c_1/', 'b/c_1/op:0')

            with tf.name_scope('b') as ns:
                self._check_ns(ns, 'b_1/', 'b_1/op:0')

    def test_variable_scope(self):
        root = tf.get_variable_scope()

        with tf.variable_scope(''):
            self._check_vs('v0', '', '', 'v0:0', 'op:0')

        with tf.variable_scope('a') as a:
            self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a/op:0')

            with tf.variable_scope('b'):
                self._check_vs('v2', 'a/b', 'a/b/', 'a/b/v2:0', 'a/b/op:0')

                with tf.variable_scope(a):
                    # having the name scope 'a/b/a' is an absurd behavior
                    # of `tf.variable_scope`, which we may not agree but
                    # have to follow.
                    self._check_vs('v3', 'a', 'a/', 'a/v3:0', 'a/b/a/op:0')

            with tf.variable_scope('b'):
                self._check_vs('v5', 'a/b', 'a/b_1/', 'a/b/v5:0',
                               'a/b_1/op:0')

        with tf.variable_scope('a/b'):
                self._check_vs('v9', 'a/b', 'a/b_2/', 'a/b/v9:0',
                               'a/b_2/op:0')

        with tf.variable_scope('a'):
            self._check_vs('v6', 'a', 'a_1/', 'a/v6:0', 'a_1/op:0')

            with tf.variable_scope('b'):
                self._check_vs('v7', 'a/b', 'a_1/b/', 'a/b/v7:0',
                               'a_1/b/op:0')

            with tf.variable_scope(None, default_name='b'):
                self._check_vs('v8', 'a/b_1', 'a_1/b_1/', 'a/b_1/v8:0',
                               'a_1/b_1/op:0')

        with tf.variable_scope(a):
            self._check_vs('v9', 'a', 'a/', 'a/v9:0', 'a_2/op:0')

            # test return to root scope
            with tf.variable_scope(root):
                self._check_vs('v10', '', '', 'v10:0', 'a_2/op_1:0')

    def test_reuse(self):
        root = tf.get_variable_scope()
        self.assertFalse(root.reuse)

        with tf.variable_scope('a', reuse=True) as a:
            self.assertTrue(a.reuse)

            with tf.variable_scope('b') as b:
                self.assertTrue(b.reuse)

            # TensorFlow >= 1.1.0 does not allow to cancel reuse flag
            if is_tensorflow_version_higher_or_equal('1.1.0'):
                with tf.variable_scope('b', reuse=False) as vs:
                    self.assertTrue(vs.reuse)

                with tf.variable_scope(b, reuse=False) as vs:
                    self.assertTrue(vs.reuse)

            # Reopen a stored variable scope will restore its reuse flag.
            with tf.variable_scope(root) as vs:
                self.assertFalse(vs.reuse)


class ReopenVariableScopeTestCase(tf.test.TestCase, _VarScopeTestMixin):

    def test_basic(self):
        root = tf.get_variable_scope()

        with tf.variable_scope('a') as a:
            self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a/op:0')

            with reopen_variable_scope(root):
                self._check_vs('v2', '', '', 'v2:0', 'op:0')

                with reopen_variable_scope(a):
                    self._check_vs('v3', 'a', 'a/', 'a/v3:0', 'a/op_1:0')

        with tf.variable_scope('a/b') as b:
            self._check_vs('v4', 'a/b', 'a/b/', 'a/b/v4:0', 'a/b/op:0')

            with reopen_variable_scope(root):
                self._check_vs('v5', '', '', 'v5:0', 'op_1:0')

            with reopen_variable_scope(a):
                self._check_vs('v6', 'a', 'a/', 'a/v6:0', 'a/op_2:0')

                with reopen_variable_scope(a):
                    self._check_vs('v7', 'a', 'a/', 'a/v7:0', 'a/op_3:0')

        with reopen_variable_scope(b):
            self._check_vs('v8', 'a/b', 'a/b/', 'a/b/v8:0', 'a/b/op_1:0')

    def test_errors(self):
        with pytest.raises(TypeError, message='`var_scope` must be an instance '
                                              'of `tf.VariableScope`'):
            with reopen_variable_scope(object()):
                pass


class RootVariableScopeTestCase(tf.test.TestCase, _VarScopeTestMixin):

    def test_root_variable_scope(self):
        with root_variable_scope() as root:
            self._check_vs('v1', '', '', 'v1:0', 'op:0')
            with tf.variable_scope('a'):
                self._check_vs('v2', 'a', 'a/', 'a/v2:0', 'a/op:0')

                with root_variable_scope():
                    self._check_vs('v3', '', '', 'v3:0', 'op_1:0')

                self._check_vs('v4', 'a', 'a/', 'a/v4:0', 'a/op_1:0')

        with root_variable_scope():
            self._check_vs('v5', '', '', 'v5:0', 'op_2:0')

        with tf.variable_scope('b'):
            with tf.variable_scope(root):
                self._check_vs('v6', '', '', 'v6:0', 'b/op:0')


class MyVarScopeObj(VarScopeObject):
    @instance_reuse
    def f(self):
        return tf.get_variable('var', shape=()), tf.add(1, 2, name='op')

    @instance_reuse
    def g(self):
        a = tf.get_variable('a', shape=())
        with tf.variable_scope('nested'):
            b = tf.get_variable('b', shape=())
            c = tf.get_variable('c', shape=(), trainable=False)
        return a, b, c


class VarScopeObjectTestCase(tf.test.TestCase):

    def test_repr(self):
        self.assertEqual(repr(MyVarScopeObj(name='a')), "MyVarScopeObj('a')")
        with tf.variable_scope('parent'):
            self.assertEqual(repr(MyVarScopeObj(scope='b')),
                             "MyVarScopeObj('parent/b')")

    def test_default_name(self):
        self.assertEqual(MyVarScopeObj().variable_scope.name,
                         'my_var_scope_obj')

    def test_name_and_scope(self):
        o1 = MyVarScopeObj(name='o')
        self.assertEqual(o1.name, 'o')
        self.assertEqual(o1.variable_scope.name, 'o')
        self.assertEqual(o1.variable_scope.original_name_scope, 'o/')
        var_1, op_1 = o1.f()
        self.assertEqual(var_1.name, 'o/f/var:0')
        self.assertEqual(op_1.name, 'o/f/op:0')

        with tf.variable_scope('child'):
            var_child, op_child = o1.f()
            self.assertIs(var_child, var_1)
            self.assertEqual(var_child.name, 'o/f/var:0')
            self.assertEqual(op_child.name, 'o/f_1/op:0')

        # test the second object with the same default name
        o2 = MyVarScopeObj(name='o')
        self.assertEqual(o2.name, 'o')
        self.assertEqual(o2.variable_scope.name, 'o_1')
        self.assertEqual(o2.variable_scope.original_name_scope, 'o_1/')
        var_2, op_2 = o2.f()
        self.assertEqual(var_2.name, 'o_1/f/var:0')
        self.assertEqual(op_2.name, 'o_1/f/op:0')

        # test the third object with the same scope as o1
        o3 = MyVarScopeObj(scope='o')
        self.assertIsNone(o3.name)
        self.assertEqual(o3.variable_scope.name, 'o')
        self.assertEqual(o3.variable_scope.original_name_scope, 'o_2/')
        var_3, op_3 = o3.f()
        self.assertEqual(var_3.name, 'o/f/var:0')
        self.assertEqual(op_3.name, 'o_2/f/op:0')

        # test the object under other scope
        with tf.variable_scope('c'):
            o4 = MyVarScopeObj(name='o')
            self.assertEqual(o4.name, 'o')
            self.assertEqual(o4.variable_scope.name, 'c/o')
            self.assertEqual(o4.variable_scope.original_name_scope, 'c/o/')
            var_4, op_4 = o4.f()
            self.assertEqual(var_4.name, 'c/o/f/var:0')
            self.assertEqual(op_4.name, 'c/o/f/op:0')

    def test_get_variables_as_dict(self):
        _ = tf.get_variable('global_var', shape=(), dtype=tf.int32)
        obj = MyVarScopeObj()
        f_var = obj.f()[0]
        a, b, c = obj.g()

        # test get all variables
        self.assertEqual(
            obj.get_variables_as_dict(),
            {'f/var': f_var, 'g/a': a, 'g/nested/b': b, 'g/nested/c': c}
        )

        # test get trainable variables
        self.assertEqual(
            obj.get_variables_as_dict(
                collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            {'f/var': f_var, 'g/a': a, 'g/nested/b': b}
        )

        # test get nested variables
        self.assertEqual(
            obj.get_variables_as_dict(sub_scope='g'),
            {'a': a, 'nested/b': b, 'nested/c': c}
        )

        # test get nested and nested variables
        self.assertEqual(
            obj.get_variables_as_dict(sub_scope='g/nested'),
            {'b': b, 'c': c}
        )

        # test get nested and nested variables with informal sub-scope name
        self.assertEqual(
            obj.get_variables_as_dict(sub_scope='/g/nested/'),
            {'b': b, 'c': c}
        )

        # test get nested and trainable variables
        self.assertEqual(
            obj.get_variables_as_dict(
                sub_scope='g', collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            {'a': a, 'nested/b': b}
        )

        # test get nested variables without removing common prefix of sub-scope
        self.assertEqual(
            obj.get_variables_as_dict(sub_scope='g/nested',
                                      strip_sub_scope=False),
            {'g/nested/b': b, 'g/nested/c': c}
        )
