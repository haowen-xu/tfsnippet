import tensorflow as tf

from tfsnippet.modules import Module


class _MyModule(Module):

    def _forward(self, inputs):
        with tf.variable_scope(None, default_name='a'):
            v1 = tf.get_variable('var', shape=())
        with tf.variable_scope(None, default_name='a'):
            v2 = tf.get_variable('var', shape=())
        return v1, v2, tf.add(v1, v2, name='op')


class ModuleTestCase(tf.test.TestCase):

    def test_construction_with_name(self):
        c = _MyModule(name='comp')
        v1, v2, op = c(None)
        self.assertEqual(v1.name, 'comp/a/var:0')
        self.assertEqual(v2.name, 'comp/a_1/var:0')
        self.assertEqual(op.name, 'comp/forward/op:0')
        v1_1, v2_1, op_1 = c(None)
        self.assertIs(v1_1, v1)
        self.assertIs(v2_1, v2)
        self.assertEqual(op_1.name, 'comp/forward_1/op:0')

        c2 = _MyModule(name='comp')
        v1_2, v2_2, op_2 = c2(None)
        self.assertEqual(v1_2.name, 'comp_1/a/var:0')
        self.assertEqual(v2_2.name, 'comp_1/a_1/var:0')
        self.assertEqual(op_2.name, 'comp_1/forward/op:0')
        self.assertIsNot(v1_2, v1)
        self.assertIsNot(v2_2, v2)

    def test_construction_in_nested_scope(self):
        c = _MyModule(name='comp')
        with tf.variable_scope('child'):
            v1, v2, op = c(None)
            self.assertEqual(v1.name, 'comp/a/var:0')
            self.assertEqual(v2.name, 'comp/a_1/var:0')
            self.assertEqual(op.name, 'comp/forward/op:0')

    def test_construction_with_scope(self):
        c = _MyModule(scope='comp')
        v1, v2, op = c(None)
        self.assertEqual(v1.name, 'comp/a/var:0')
        self.assertEqual(v2.name, 'comp/a_1/var:0')
        self.assertEqual(op.name, 'comp/forward/op:0')

        c2 = _MyModule(scope='comp')
        v1_1, v2_1, op_1 = c2(None)
        self.assertEqual(v1_1.name, 'comp/a/var:0')
        self.assertEqual(v2_1.name, 'comp/a_1/var:0')
        self.assertEqual(op_1.name, 'comp_1/forward/op:0')
        self.assertIs(v1_1, v1)
        self.assertIs(v2_1, v2)
