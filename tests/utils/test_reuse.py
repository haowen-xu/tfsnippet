import pytest
import tensorflow as tf

from tfsnippet.utils import global_reuse


class GlobalReuseTestCase(tf.test.TestCase):

    def test_create_in_root_scope(self):
        @global_reuse('the_scope')
        def make_var_and_op():
            vs = tf.get_variable_scope()
            var = tf.get_variable('var', shape=(), dtype=tf.float32)
            op = tf.add(1, 2, name='op')
            return vs, var, op

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
            vs = tf.get_variable_scope()
            var = tf.get_variable('var', shape=(), dtype=tf.float32)
            op = tf.add(1, 2, name='op')
            return vs, var, op

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

    def test_create_variable_scope_with_default_name_within_global_reuse(self):
        @global_reuse('the_scope')
        def make_variable_scopes():
            with tf.variable_scope(None, default_name='vs') as vs1:
                var1 = tf.get_variable('var', shape=(), dtype=tf.float32)
            with tf.variable_scope(None, default_name='vs') as vs2:
                var2 = tf.get_variable('var', shape=(), dtype=tf.float32)

            return (vs1, var1), (vs2, var2)

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

    def test_work_with_tf_make_template(self):
        @global_reuse('the_scope')
        def make_template():
            def f():
                vs = tf.get_variable_scope()
                var = tf.get_variable('var', shape=(), dtype=tf.float32)
                op = tf.add(1, 2, name='op')
                return vs, var, op

            def g():
                with tf.variable_scope(None, default_name='vs') as vs1:
                    var1 = tf.get_variable('var', dtype=tf.float32, shape=())
                with tf.variable_scope(None, default_name='vs') as vs2:
                    var2 = tf.get_variable('var', dtype=tf.float32, shape=())
                return (vs1, var1), (vs2, var2)

            f = tf.make_template(
                'f', f, unique_name_='f', create_scope_now_=True)
            g = tf.make_template(
                'g', g, unique_name_='g', create_scope_now_=True)
            return f, g

        with tf.Graph().as_default():
            # test make the template under root scope
            f, g = make_template()
            vs, var, op = f()
            self.assertEqual(vs.name, 'the_scope/f')
            self.assertEqual(var.name, 'the_scope/f/var:0')
            self.assertEqual(op.name, 'f/op:0')
            (vs1, var1), (vs2, var2) = g()
            self.assertEqual(vs1.name, 'the_scope/g/vs')
            self.assertEqual(var1.name, 'the_scope/g/vs/var:0')
            self.assertEqual(vs2.name, 'the_scope/g/vs_1')
            self.assertEqual(var2.name, 'the_scope/g/vs_1/var:0')

            # now call the template function for the second time
            vs, var, op = f()
            self.assertEqual(vs.name, 'the_scope/f')
            self.assertEqual(var.name, 'the_scope/f/var:0')
            self.assertEqual(op.name, 'f_1/op:0')
            (vs1, var1), (vs2, var2) = g()
            self.assertEqual(vs1.name, 'the_scope/g/vs')
            self.assertEqual(var1.name, 'the_scope/g/vs/var:0')
            self.assertEqual(vs2.name, 'the_scope/g/vs_1')
            self.assertEqual(var2.name, 'the_scope/g/vs_1/var:0')

            # call the template function under another variable scope
            with tf.variable_scope('parent'):
                vs, var, op = f()
                self.assertEqual(vs.name, 'the_scope/f')
                self.assertEqual(var.name, 'the_scope/f/var:0')
                self.assertEqual(op.name, 'parent/f/op:0')
                (vs1, var1), (vs2, var2) = g()
                self.assertEqual(vs1.name, 'the_scope/g/vs')
                self.assertEqual(var1.name, 'the_scope/g/vs/var:0')
                self.assertEqual(vs2.name, 'the_scope/g/vs_1')
                self.assertEqual(var2.name, 'the_scope/g/vs_1/var:0')

            # test to obtain the template under another variable scope
            with tf.variable_scope('outside'):
                f, g = make_template()
                vs, var, op = f()
                self.assertEqual(vs.name, 'the_scope/f')
                self.assertEqual(var.name, 'the_scope/f/var:0')
                self.assertEqual(op.name, 'outside/f/op:0')
                (vs1, var1), (vs2, var2) = g()
                self.assertEqual(vs1.name, 'the_scope/g/vs')
                self.assertEqual(var1.name, 'the_scope/g/vs/var:0')
                self.assertEqual(vs2.name, 'the_scope/g/vs_1')
                self.assertEqual(var2.name, 'the_scope/g/vs_1/var:0')

            # now test to call the template under root scope
            vs, var, op = f()
            self.assertEqual(vs.name, 'the_scope/f')
            self.assertEqual(var.name, 'the_scope/f/var:0')
            self.assertEqual(op.name, 'f_2/op:0')
            (vs1, var1), (vs2, var2) = g()
            self.assertEqual(vs1.name, 'the_scope/g/vs')
            self.assertEqual(var1.name, 'the_scope/g/vs/var:0')
            self.assertEqual(vs2.name, 'the_scope/g/vs_1')
            self.assertEqual(var2.name, 'the_scope/g/vs_1/var:0')

    def test_nested_name_cause_an_error(self):
        with pytest.raises(ValueError, match='`global_reuse` does not support '
                                             '"/" in scope name'):
            @global_reuse('nested/scope')
            def nested_scope():
                return tf.get_variable('var', shape=(), dtype=tf.float32)

    def test_auto_choose_scope_name(self):
        @global_reuse
        def f():
            vs = tf.get_variable_scope()
            var = tf.get_variable('var', shape=(), dtype=tf.float32)
            return vs, var

        with tf.Graph().as_default():
            vs, var = f()
            self.assertEqual(vs.name, 'f')
            self.assertEqual(var.name, 'f/var:0')

    def test_uniquified_scope_name(self):
        @global_reuse('the_scope')
        def f1():
            vs = tf.get_variable_scope()
            var = tf.get_variable('var', shape=(), dtype=tf.float32)
            return vs, var

        @global_reuse('the_scope')
        def f2():
            vs = tf.get_variable_scope()
            var = tf.get_variable('var', shape=(), dtype=tf.float32)
            return vs, var

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
            vs = tf.get_variable_scope()
            var = tf.get_variable('var', shape=(), dtype=tf.float32)
            return vs, var

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
