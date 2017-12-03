import pytest
import tensorflow as tf

from tfsnippet.scaffold import auto_reuse_variables, instance_reuse


class AutoReuseVariablesTestCase(tf.test.TestCase):

    def _check_vs(self, get_var_name, vs_name, vs_scope, var_name, op_name):
        vs = tf.get_variable_scope()
        self.assertEqual(vs.name, vs_name)
        self.assertEqual(vs.original_name_scope, vs_scope)
        var = tf.get_variable(get_var_name, shape=())
        self.assertEqual(var.name, var_name)
        self.assertEqual(tf.add(1, 2, name='op').name, op_name)
        return var

    def test_basic_reuse(self):
        with auto_reuse_variables('a') as a:
            self.assertFalse(a.reuse)
            v1 = self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a/op:0')

        with auto_reuse_variables('a') as vs:
            self.assertTrue(vs.reuse)
            v1_2 = self._check_vs('v1', 'a', 'a_1/', 'a/v1:0', 'a_1/op:0')
            self.assertIs(v1_2, v1)

            with pytest.raises(
                    ValueError,
                    message='Variable a/v2 does not exist, or was not '
                            'created with tf.get_variable()'):
                tf.get_variable('v2', shape=())

        with auto_reuse_variables(a):
            self.assertTrue(vs.reuse)
            v1_3 = self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a_2/op:0')
            self.assertIs(v1_3, v1)

    def test_nested_reuse(self):
        with auto_reuse_variables('a') as a:
            self.assertFalse(a.reuse)

            with auto_reuse_variables('b') as b:
                self.assertFalse(b.reuse)
                b1 = self._check_vs('v1', 'a/b', 'a/b/', 'a/b/v1:0',
                                    'a/b/op:0')

            with auto_reuse_variables('b') as vs:
                self.assertTrue(vs.reuse)
                b1_2 = self._check_vs('v1', 'a/b', 'a/b_1/', 'a/b/v1:0',
                                      'a/b_1/op:0')
                self.assertIs(b1_2, b1)

            with auto_reuse_variables(b) as vs:
                self.assertTrue(vs.reuse)
                b1_3 = self._check_vs('v1', 'a/b', 'a/b/', 'a/b/v1:0',
                                      'a/b_2/op:0')
                self.assertIs(b1_3, b1)

        with auto_reuse_variables('a/b') as vs:
            self.assertTrue(vs.reuse)
            b1_4 = self._check_vs('v1', 'a/b', 'a/b_3/', 'a/b/v1:0',
                                  'a/b_3/op:0')
            self.assertIs(b1_4, b1)

        with auto_reuse_variables(b) as vs:
            self.assertTrue(vs.reuse)
            # having the name scope 'b' is an absurd behavior
            # of `tf.variable_scope`, which we may not agree but
            # have to follow.
            b1_5 = self._check_vs('v1', 'a/b', 'a/b/', 'a/b/v1:0',
                                  'b/op:0')
            self.assertIs(b1_5, b1)

    def test_mix_reuse_and_variable_scope(self):
        with tf.variable_scope('a') as a:
            self.assertFalse(a.reuse)

            with auto_reuse_variables('b') as b:
                self.assertFalse(b.reuse)
                b1 = self._check_vs('v1', 'a/b', 'a/b/', 'a/b/v1:0',
                                    'a/b/op:0')

            with auto_reuse_variables('b') as vs:
                self.assertTrue(vs.reuse)
                b1_2 = self._check_vs('v1', 'a/b', 'a/b_1/', 'a/b/v1:0',
                                      'a/b_1/op:0')
                self.assertIs(b1_2, b1)

        with auto_reuse_variables('a') as vs:
            self.assertFalse(vs.reuse)

            with auto_reuse_variables('b') as vs:
                self.assertTrue(vs.reuse)
                b1_3 = self._check_vs('v1', 'a/b', 'a_1/b/', 'a/b/v1:0',
                                      'a_1/b/op:0')
                self.assertIs(b1_3, b1)

    def test_reopen_name_scope(self):
        with auto_reuse_variables('a') as a:
            self.assertFalse(a.reuse)
            v1 = self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a/op:0')

        with auto_reuse_variables(a, reopen_name_scope=True) as vs:
            self.assertTrue(vs.reuse)
            v1_2 = self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a/op_1:0')
            self.assertIs(v1_2, v1)

            with pytest.raises(
                    ValueError,
                    message='Variable a/v2 does not exist, or was not '
                            'created with tf.get_variable()'):
                tf.get_variable('v2', shape=())

    def test_different_graph(self):
        with tf.Graph().as_default():
            with auto_reuse_variables('a') as a:
                self.assertFalse(a.reuse)
                v1 = self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a/op:0')

            with auto_reuse_variables('a') as vs:
                self.assertTrue(vs.reuse)
                v1_2 = self._check_vs('v1', 'a', 'a_1/', 'a/v1:0', 'a_1/op:0')
                self.assertIs(v1_2, v1)

        with tf.Graph().as_default():
            with auto_reuse_variables('a') as vs:
                self.assertFalse(vs.reuse)
                v1_3 = self._check_vs('v1', 'a', 'a/', 'a/v1:0', 'a/op:0')
                self.assertIsNot(v1_3, v1)

    def test_reuse_root(self):
        root = tf.get_variable_scope()

        with auto_reuse_variables(root) as vs:
            self.assertFalse(vs.reuse)
            v0 = self._check_vs('v0', '', '', 'v0:0', 'op:0')

        with tf.variable_scope('a'):
            with auto_reuse_variables(root) as vs:
                self.assertTrue(vs.reuse)
                v0_1 = self._check_vs('v0', '', '', 'v0:0', 'a/op:0')
                self.assertIs(v0_1, v0)

    def test_errors(self):
        with pytest.raises(ValueError, match='`name_or_scope` cannot be empty'):
            with auto_reuse_variables(''):
                pass

        with pytest.raises(ValueError, match='`name_or_scope` cannot be empty'):
            with auto_reuse_variables(None):
                pass

        with pytest.raises(
                ValueError, message='`reopen_name_scope` can be set to True '
                                    'only if `name_or_scope` is an instance of '
                                    '`tf.VariableScope`'):
            with auto_reuse_variables('a', reopen_name_scope=True):
                pass


class InstanceReuseTestCase(tf.test.TestCase):

    def test_basic(self):
        class _Reusable(object):
            def __init__(self, name):
                with tf.variable_scope(None, default_name=name) as vs:
                    self.variable_scope = vs

            @instance_reuse
            def f(self):
                return tf.get_variable('var', shape=()), tf.add(1, 2, name='op')

            @instance_reuse(scope='f')
            def f_1(self):
                return tf.get_variable('var', shape=())

            @instance_reuse
            def g(self):
                return self.f()

        obj = _Reusable('obj')

        # test reuse with default settings
        var_1, op_1 = obj.f()
        var_2, op_2 = obj.f()
        self.assertIs(var_1, var_2)
        self.assertIsNot(op_1, op_2)
        self.assertEqual(var_1.name, 'obj/f/var:0')
        self.assertEqual(op_1.name, 'obj/f/op:0')
        self.assertEqual(op_2.name, 'obj/f_1/op:0')

        # test reuse according to the scope name
        var_3 = obj.f_1()
        self.assertIs(var_3, var_1)

        # test reuse within different parent scope
        with tf.variable_scope('parent'):
            var_3, op_3 = obj.f()
            self.assertIs(var_3, var_1)
            self.assertIsNot(op_3, op_1)
            self.assertEqual(var_3.name, 'obj/f/var:0')
            self.assertEqual(op_3.name, 'obj/f_3/op:0')

        # test reuse with nested call
        var_4, op_4 = obj.g()
        var_5, op_5 = obj.g()
        self.assertIs(var_4, var_5)
        self.assertIs(var_4, var_1)
        self.assertIsNot(op_4, op_5)
        self.assertEqual(var_4.name, 'obj/f/var:0')
        self.assertEqual(op_4.name, 'obj/f_4/op:0')
        self.assertEqual(op_5.name, 'obj/f_5/op:0')

        # test reuse on another object
        obj2 = _Reusable('obj')
        var_6, op_6 = obj2.f()
        var_7, op_7 = obj2.f()
        self.assertIs(var_6, var_7)
        self.assertIsNot(var_6, var_1)
        self.assertIsNot(op_6, op_7)
        self.assertEqual(var_6.name, 'obj_1/f/var:0')
        self.assertEqual(op_6.name, 'obj_1/f/op:0')
        self.assertEqual(op_7.name, 'obj_1/f_1/op:0')

    def test_nested_scope(self):
        class _Reusable(object):
            def __init__(self, name):
                with tf.variable_scope(None, default_name=name) as vs:
                    self.variable_scope = vs

            @instance_reuse(scope='nested/scope')
            def nested(self):
                return tf.get_variable('var', shape=()), tf.add(1, 2, name='op')

        obj = _Reusable('obj')
        var_1, op_1 = obj.nested()
        var_2, op_2 = obj.nested()
        self.assertIs(var_1, var_2)
        self.assertIsNot(op_1, op_2)
        self.assertEqual(var_1.name, 'obj/nested/scope/var:0')
        self.assertEqual(op_1.name, 'obj/nested/scope/op:0')
        self.assertEqual(op_2.name, 'obj/nested/scope_1/op:0')

    def test_errors(self):
        class _Reusable(object):
            def __init__(self):
                self.variable_scope = ''

            @instance_reuse
            def f(self):
                pass

        obj = _Reusable()
        with pytest.raises(
                TypeError,
                match='`variable_scope` attribute of the instance .* '
                      'is expected to be a `tf.VariableScope`.*'):
            obj.f()

        with pytest.raises(
                TypeError,
                match='`method` seems not to be an instance method.*'):
            @instance_reuse
            def f():
                pass

        with pytest.raises(
                TypeError,
                match='`method` seems not to be an instance method.*'):
            @instance_reuse
            def f(a):
                pass

        with pytest.raises(
                TypeError,
                message='`method` is expected to be unbound instance method'):
            obj = _Reusable()
            instance_reuse(obj.f)
