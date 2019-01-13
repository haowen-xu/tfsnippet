import pytest
import tensorflow as tf

from tfsnippet.shortcuts import *
from tfsnippet.utils import *


def _make_var_and_op():
    vs = tf.get_variable_scope()
    assert(vs.name == get_reuse_stack_top().name)
    var = tf.get_variable('var', shape=(), dtype=tf.float32)
    op = tf.add(1, 2, name='op')
    return vs, var, op


def _make_variable_scope():
    vs = tf.get_variable_scope()
    assert(vs.name == get_reuse_stack_top().name)
    var = tf.get_variable('var', shape=(), dtype=tf.float32)
    return vs, var


def _make_variable_scopes():
    vs = tf.get_variable_scope()
    assert(vs.name == get_reuse_stack_top().name)
    with tf.variable_scope(None, default_name='vs') as vs1:
        var1 = tf.get_variable('var', shape=(), dtype=tf.float32)
    with tf.variable_scope(None, default_name='vs') as vs2:
        var2 = tf.get_variable('var', shape=(), dtype=tf.float32)
    return (vs1, var1), (vs2, var2)


class InstanceReuseTestCase(tf.test.TestCase):

    def test_errors(self):
        class _Reusable(object):
            def __init__(self):
                self.variable_scope = ''

            @instance_reuse('foo')
            def f(self):
                pass

        obj = _Reusable()
        with pytest.raises(TypeError, match='`variable_scope` attribute of '
                                            'the instance .* is expected to '
                                            'be a `tf.VariableScope`.*'):
            obj.f()

        with pytest.raises(TypeError, match='`method` seems not to be an '
                                            'instance method.*'):
            @instance_reuse('foo')
            def f():
                pass

        with pytest.raises(TypeError, match='`method` seems not to be an '
                                            'instance method.*'):
            @instance_reuse('foo')
            def f(a):
                pass

        with pytest.raises(TypeError, match='`method` is expected to be '
                                            'unbound instance method'):
            obj = _Reusable()
            instance_reuse(obj.f)

    def test_nested_name_should_cause_an_error(self):
        with pytest.raises(ValueError,
                           match='`instance_reuse` does not support "/" in '
                                 'scope name'):
            class MyScopeObject(VarScopeObject):
                @instance_reuse('nested/scope')
                def foo(self):
                    return _make_var_and_op()

    def test_create_in_root_scope(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse('foo')
            def foo(self):
                return _make_var_and_op()

        with tf.Graph().as_default():
            o = MyScopeObject('o')

            # test enter for the first time
            vs, var, op = o.foo()
            self.assertEqual(vs.name, 'o/foo')
            self.assertEqual(var.name, 'o/foo/var:0')
            # YES! THIS IS THE EXPECTED BEHAVIOR!
            self.assertEqual(op.name, 'foo/op:0')

            # test enter for the second time
            vs, var, op = o.foo()
            self.assertEqual(vs.name, 'o/foo')
            self.assertEqual(var.name, 'o/foo/var:0')
            self.assertEqual(op.name, 'foo_1/op:0')

            # now we enter a variable scope, and then call `foo` twice.
            with tf.variable_scope('parent'):
                # call the method for the first time
                vs, var, op = o.foo()
                self.assertEqual(vs.name, 'o/foo')
                self.assertEqual(var.name, 'o/foo/var:0')
                self.assertEqual(op.name, 'parent/foo/op:0')

                # call the method for the second time
                vs, var, op = o.foo()
                self.assertEqual(vs.name, 'o/foo')
                self.assertEqual(var.name, 'o/foo/var:0')
                self.assertEqual(op.name, 'parent/foo_1/op:0')

    def test_create_in_parent_scope(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse('foo')
            def foo(self):
                return _make_var_and_op()

        with tf.Graph().as_default():
            with tf.variable_scope('parent'):
                o = MyScopeObject('o')

            # test enter for the first time
            vs, var, op = o.foo()
            self.assertEqual(vs.name, 'parent/o/foo')
            self.assertEqual(var.name, 'parent/o/foo/var:0')
            # YES! THIS IS THE EXPECTED BEHAVIOR!
            self.assertEqual(op.name, 'foo/op:0')

            # test enter for the second time
            vs, var, op = o.foo()
            self.assertEqual(vs.name, 'parent/o/foo')
            self.assertEqual(var.name, 'parent/o/foo/var:0')
            self.assertEqual(op.name, 'foo_1/op:0')

            # now we enter a variable scope, and then call `foo` twice.
            with tf.variable_scope('another'):
                # call the method for the first time
                vs, var, op = o.foo()
                self.assertEqual(vs.name, 'parent/o/foo')
                self.assertEqual(var.name, 'parent/o/foo/var:0')
                self.assertEqual(op.name, 'another/foo/op:0')

                # call the method for the second time
                vs, var, op = o.foo()
                self.assertEqual(vs.name, 'parent/o/foo')
                self.assertEqual(var.name, 'parent/o/foo/var:0')
                self.assertEqual(op.name, 'another/foo_1/op:0')

    def test_call_in_original_name_scope(self):
        class MyScopeObject(VarScopeObject):

            def __init__(self, *args, **kwargs):
                super(MyScopeObject, self).__init__(*args, **kwargs)
                with reopen_variable_scope(self.variable_scope):
                    self.vs, self.var, self.op = self.foo()

            @instance_reuse('foo')
            def foo(self):
                return _make_var_and_op()

        with tf.Graph().as_default():
            o = MyScopeObject('o')

            # call within _variable_scope_created should not generate
            # a new name scope.
            self.assertEqual(o.vs.name, 'o/foo')
            self.assertEqual(o.var.name, 'o/foo/var:0')
            self.assertEqual(o.op.name, 'o/foo/op:0')

            # call it for the second time within the object's variable scope
            # and the object's original name scope (this actually will not
            # happen in a regular program).
            with tf.variable_scope(o.variable_scope,
                                   auxiliary_name_scope=False):
                with tf.name_scope(o.variable_scope.original_name_scope):
                    vs, var, op = o.foo()
                    self.assertEqual(vs.name, 'o/foo')
                    self.assertEqual(var.name, 'o/foo/var:0')
                    self.assertEqual(op.name, 'o/foo_1/op:0')

    def test_create_variable_scopes_with_default_name(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse('foo')
            def foo(self):
                return _make_variable_scopes()

        with tf.Graph().as_default():
            o = MyScopeObject('o')

            # test enter for the first time
            (vs1, var1), (vs2, var2) = o.foo()
            self.assertEqual(vs1.name, 'o/foo/vs')
            self.assertEqual(var1.name, 'o/foo/vs/var:0')
            self.assertEqual(vs2.name, 'o/foo/vs_1')
            self.assertEqual(var2.name, 'o/foo/vs_1/var:0')

            # test enter for the second time, should reuse the variables
            (vs1, var1), (vs2, var2) = o.foo()
            self.assertEqual(vs1.name, 'o/foo/vs')
            self.assertEqual(var1.name, 'o/foo/vs/var:0')
            self.assertEqual(vs2.name, 'o/foo/vs_1')
            self.assertEqual(var2.name, 'o/foo/vs_1/var:0')

    def test_auto_choose_scope_name(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse
            def foo(self):
                return _make_variable_scope()

        with tf.Graph().as_default():
            o = MyScopeObject('o')
            vs, var = o.foo()
            self.assertEqual(vs.name, 'o/foo')
            self.assertEqual(var.name, 'o/foo/var:0')

    def test_auto_choose_scope_name_2(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse()
            def foo(self):
                return _make_variable_scope()

        with tf.Graph().as_default():
            o = MyScopeObject('o')
            vs, var = o.foo()
            self.assertEqual(vs.name, 'o/foo')
            self.assertEqual(var.name, 'o/foo/var:0')

    def test_non_uniquified_scope_name(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse('foo')
            def foo_1(self):
                return tf.get_variable('bar', shape=())

            @instance_reuse('foo')
            def foo_2(self):
                return tf.get_variable('bar', shape=())

            @instance_reuse('foo')
            def foo_3(self):
                return tf.get_variable('bar2', shape=())

        with tf.Graph().as_default():
            o = MyScopeObject('o')

            self.assertEqual(o.foo_1().name, 'o/foo/bar:0')
            with pytest.raises(ValueError, match='Variable o/foo/bar already '
                                                 'exists'):
                _ = o.foo_2()
            self.assertEqual(o.foo_3().name, 'o/foo/bar2:0')

    def test_two_instances(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse
            def foo(self):
                return _make_variable_scope()

        with tf.Graph().as_default():
            o1 = MyScopeObject('o1')
            vs1, var1 = o1.foo()
            self.assertEqual(vs1.name, 'o1/foo')
            self.assertEqual(var1.name, 'o1/foo/var:0')

            o2 = MyScopeObject('o2')
            vs2, var2 = o2.foo()
            self.assertEqual(vs2.name, 'o2/foo')
            self.assertEqual(var2.name, 'o2/foo/var:0')


class GlobalReuseTestCase(tf.test.TestCase):

    def test_nested_name_should_cause_an_error(self):
        with pytest.raises(ValueError, match='`global_reuse` does not support '
                                             '"/" in scope name'):
            @global_reuse('nested/scope')
            def nested_scope():
                return tf.get_variable('var', shape=(), dtype=tf.float32)

    def test_create_in_root_scope(self):
        @global_reuse('the_scope')
        def make_var_and_op():
            return _make_var_and_op()

        with tf.Graph().as_default():
            # test enter for the first time
            vs, var, op = make_var_and_op()
            self.assertEqual(vs.name, 'the_scope')
            self.assertEqual(var.name, 'the_scope/var:0')
            self.assertEqual(op.name, 'the_scope/op:0')

            # enter for the second time
            vs, var, op = make_var_and_op()
            self.assertEqual(vs.name, 'the_scope')
            self.assertEqual(var.name, 'the_scope/var:0')
            self.assertEqual(op.name, 'the_scope_1/op:0')

            # now we enter a variable scope, and then call the method twice.
            with tf.variable_scope('parent'):
                # call the method for the first time
                vs, var, op = make_var_and_op()
                self.assertEqual(vs.name, 'the_scope')
                self.assertEqual(var.name, 'the_scope/var:0')
                self.assertEqual(op.name, 'parent/the_scope/op:0')

                # call the method for the second time
                vs, var, op = make_var_and_op()
                self.assertEqual(vs.name, 'the_scope')
                self.assertEqual(var.name, 'the_scope/var:0')
                self.assertEqual(op.name, 'parent/the_scope_1/op:0')

    def test_create_in_parent_scope(self):
        @global_reuse('the_scope')
        def make_var_and_op():
            return _make_var_and_op()

        with tf.Graph().as_default():
            # open the parent scope
            with tf.variable_scope('parent'):
                # test enter for the first time
                vs, var, op = make_var_and_op()
                self.assertEqual(vs.name, 'the_scope')
                self.assertEqual(var.name, 'the_scope/var:0')
                self.assertEqual(op.name, 'parent/the_scope/op:0')

                # enter for the second time
                vs, var, op = make_var_and_op()
                self.assertEqual(vs.name, 'the_scope')
                self.assertEqual(var.name, 'the_scope/var:0')
                self.assertEqual(op.name, 'parent/the_scope_1/op:0')

            # now we reach the root scope, and then call the method twice.
            # call the method for the first time
            vs, var, op = make_var_and_op()
            self.assertEqual(vs.name, 'the_scope')
            self.assertEqual(var.name, 'the_scope/var:0')
            self.assertEqual(op.name, 'the_scope_1/op:0')

            # call the method for the second time
            vs, var, op = make_var_and_op()
            self.assertEqual(vs.name, 'the_scope')
            self.assertEqual(var.name, 'the_scope/var:0')
            self.assertEqual(op.name, 'the_scope_2/op:0')

    def test_create_variable_scopes_with_default_name(self):
        @global_reuse('the_scope')
        def make_variable_scopes():
            return _make_variable_scopes()

        with tf.Graph().as_default():
            # test enter for the first time
            (vs1, var1), (vs2, var2) = make_variable_scopes()
            self.assertEqual(vs1.name, 'the_scope/vs')
            self.assertEqual(var1.name, 'the_scope/vs/var:0')
            self.assertEqual(vs2.name, 'the_scope/vs_1')
            self.assertEqual(var2.name, 'the_scope/vs_1/var:0')

            # test enter for the second time, should reuse the variables
            (vs1, var1), (vs2, var2) = make_variable_scopes()
            self.assertEqual(vs1.name, 'the_scope/vs')
            self.assertEqual(var1.name, 'the_scope/vs/var:0')
            self.assertEqual(vs2.name, 'the_scope/vs_1')
            self.assertEqual(var2.name, 'the_scope/vs_1/var:0')

    def test_auto_choose_scope_name(self):
        @global_reuse
        def f():
            return _make_variable_scope()

        with tf.Graph().as_default():
            vs, var = f()
            self.assertEqual(vs.name, 'f')
            self.assertEqual(var.name, 'f/var:0')

    def test_auto_choose_scope_name_2(self):
        @global_reuse()
        def f():
            return _make_variable_scope()

        with tf.Graph().as_default():
            vs, var = f()
            self.assertEqual(vs.name, 'f')
            self.assertEqual(var.name, 'f/var:0')

    def test_uniquified_scope_name(self):
        @global_reuse('the_scope')
        def f1():
            return _make_variable_scope()

        @global_reuse('the_scope')
        def f2():
            return _make_variable_scope()

        with tf.Graph().as_default():
            vs1, var1 = f1()
            self.assertEqual(vs1.name, 'the_scope')
            self.assertEqual(var1.name, 'the_scope/var:0')

            # this function should have a different variable scope than
            # the previous function.
            vs2, var2 = f2()
            self.assertEqual(vs2.name, 'the_scope_1')
            self.assertEqual(var2.name, 'the_scope_1/var:0')

    def test_different_graph(self):
        @global_reuse('the_scope')
        def f():
            return _make_variable_scope()

        with tf.Graph().as_default() as graph1:
            vs, var1 = f()
            self.assertEqual(vs.name, 'the_scope')
            self.assertIs(var1.graph, graph1)
            self.assertEqual(var1.name, 'the_scope/var:0')

        with graph1.as_default():
            vs, var1_1 = f()
            self.assertEqual(vs.name, 'the_scope')
            self.assertIs(var1_1, var1)

        with tf.Graph().as_default() as graph2:
            vs, var2 = f()
            self.assertEqual(vs.name, 'the_scope')
            self.assertIs(var2.graph, graph2)
            self.assertEqual(var2.name, 'the_scope/var:0')


class ReuseCompatibilityTestCase(tf.test.TestCase):

    def test_instance_reuse_inside_global_reuse(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse
            def foo(self):
                return _make_var_and_op()

        @global_reuse
        def f():
            o = MyScopeObject(scope='o')
            return o, o.foo()

        with tf.Graph().as_default():
            # test call for the first time
            o, (vs, var, op) = f()
            self.assertEqual(o.variable_scope.name, 'f/o')
            self.assertEqual(vs.name, 'f/o/foo')
            self.assertEqual(var.name, 'f/o/foo/var:0')
            self.assertEqual(op.name, 'f/foo/op:0')

            # test call for the second time
            o, (vs, var, op) = f()
            self.assertEqual(o.variable_scope.name, 'f/o')
            self.assertEqual(vs.name, 'f/o/foo')
            self.assertEqual(var.name, 'f/o/foo/var:0')
            self.assertEqual(op.name, 'f_1/foo/op:0')

            # test call for the third time,
            # even it is called within another variable scope.
            # with tf.variable_scope('another'):
            with tf.variable_scope('another'):
                o, (vs, var, op) = f()
                self.assertEqual(o.variable_scope.name, 'f/o')
                self.assertEqual(vs.name, 'f/o/foo')
                self.assertEqual(var.name, 'f/o/foo/var:0')
                self.assertEqual(op.name, 'another/f/foo/op:0')

    def test_global_reuse_inside_instance_reuse(self):
        @global_reuse
        def f():
            return _make_var_and_op()

        class MyScopeObject(VarScopeObject):
            @instance_reuse
            def foo(self):
                return f(), f()

        with tf.Graph().as_default():
            o1 = MyScopeObject('o')
            o2 = MyScopeObject('o')

            # test call o1 for the first time
            (vs1, var1, op1), (vs2, var2, op2) = o1.foo()
            self.assertEqual(vs1.name, 'f')
            self.assertEqual(var1.name, 'f/var:0')
            self.assertEqual(op1.name, 'foo/f/op:0')
            self.assertEqual(vs2.name, 'f')
            self.assertEqual(var2.name, 'f/var:0')
            self.assertEqual(op2.name, 'foo/f_1/op:0')

            # test call o2 for the first time
            (vs1, var1, op1), (vs2, var2, op2) = o2.foo()
            self.assertEqual(vs1.name, 'f')
            self.assertEqual(var1.name, 'f/var:0')
            self.assertEqual(op1.name, 'foo_1/f/op:0')
            self.assertEqual(vs2.name, 'f')
            self.assertEqual(var2.name, 'f/var:0')
            self.assertEqual(op2.name, 'foo_1/f_1/op:0')

            # test call o1 for the second time,
            # even it is called within another variable scope.
            with tf.variable_scope('another'):
                (vs1, var1, op1), (vs2, var2, op2) = o1.foo()
                self.assertEqual(vs1.name, 'f')
                self.assertEqual(var1.name, 'f/var:0')
                self.assertEqual(op1.name, 'another/foo/f/op:0')
                self.assertEqual(vs2.name, 'f')
                self.assertEqual(var2.name, 'f/var:0')
                self.assertEqual(op2.name, 'another/foo/f_1/op:0')

    def test_tf_layers_inside_global_reuse(self):
        @global_reuse
        def foo():
            x = tf.zeros([2, 4], dtype=tf.float32)
            x = tf.layers.dense(x, 5)
            x = tf.layers.dense(x, 3)
            return x

        with tf.Graph().as_default():
            # We use shape -> name mapping to check the created variables,
            # because we are not sure whether or not the naming convention
            # of tf.layers will change in the future.
            _ = foo()
            shape_names = {get_static_shape(v): v.name
                           for v in tf.global_variables()}
            self.assertEqual(len(shape_names), 4)
            self.assertStartsWith(shape_names[(4, 5)], 'foo/dense/')
            self.assertStartsWith(shape_names[(5,)], 'foo/dense/')
            self.assertStartsWith(shape_names[(5, 3)], 'foo/dense_1/')
            self.assertStartsWith(shape_names[(3,)], 'foo/dense_1/')

            # Second call to the function will not add new variables,
            # even it is called within another variable scope.
            with tf.variable_scope('another'):
                _ = foo()
                shape_names = {get_static_shape(v): v.name
                               for v in tf.global_variables()}
                self.assertEqual(len(shape_names), 4)
                self.assertStartsWith(shape_names[(4, 5)], 'foo/dense/')
                self.assertStartsWith(shape_names[(5,)], 'foo/dense/')
                self.assertStartsWith(shape_names[(5, 3)], 'foo/dense_1/')
                self.assertStartsWith(shape_names[(3,)], 'foo/dense_1/')

    def test_tf_layers_inside_instance_reuse(self):
        class MyScopeObject(VarScopeObject):
            @instance_reuse
            def foo(self):
                x = tf.zeros([2, 4], dtype=tf.float32)
                x = tf.layers.dense(x, 5)
                x = tf.layers.dense(x, 3)
                return x

        with tf.Graph().as_default():
            o = MyScopeObject('o')

            # We use shape -> name mapping to check the created variables,
            # because we are not sure whether or not the naming convention
            # of tf.layers will change in the future.
            _ = o.foo()
            shape_names = {get_static_shape(v): v.name
                           for v in tf.global_variables()}
            self.assertEqual(len(shape_names), 4)
            self.assertStartsWith(shape_names[(4, 5)], 'o/foo/dense/')
            self.assertStartsWith(shape_names[(5,)], 'o/foo/dense/')
            self.assertStartsWith(shape_names[(5, 3)], 'o/foo/dense_1/')
            self.assertStartsWith(shape_names[(3,)], 'o/foo/dense_1/')

            # Second call to the function will not add new variables,
            # even it is called within another variable scope.
            with tf.variable_scope('another'):
                _ = o.foo()
                shape_names = {get_static_shape(v): v.name
                               for v in tf.global_variables()}
                self.assertEqual(len(shape_names), 4)
                self.assertStartsWith(shape_names[(4, 5)], 'o/foo/dense/')
                self.assertStartsWith(shape_names[(5,)], 'o/foo/dense/')
                self.assertStartsWith(shape_names[(5, 3)], 'o/foo/dense_1/')
                self.assertStartsWith(shape_names[(3,)], 'o/foo/dense_1/')


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

    def test_reopen_variable_scope(self):
        class MyVarScopeObj(VarScopeObject):
            def __init__(self, name=None, scope=None):
                super(MyVarScopeObj, self).__init__(name=name, scope=scope)
                with reopen_variable_scope(self.variable_scope) as vs:
                    self.vs = vs
                    self.a = tf.get_variable('a', shape=(), dtype=tf.float32)
                    self.op = tf.add(1, 2, name='op')

        o = MyVarScopeObj(name='o')
        self.assertEqual(o.vs.name, 'o')
        self.assertEqual(o.a.name, 'o/a:0')
        self.assertEqual(o.op.name, 'o/op:0')
