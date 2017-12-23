import tensorflow as tf

from tfsnippet.modules import Lambda


class LambdaTestCase(tf.test.TestCase):

    def test_scopes(self):
        def f(inputs):
            var_name, op_name = inputs
            return (
                tf.get_variable(var_name, shape=(), dtype=tf.int32),
                tf.add(1, 2, name=op_name)
            )

        comp = Lambda(f)
        var, op = comp(['var', 'op'])
        self.assertEqual(var.name, 'lambda/var:0')
        self.assertEqual(op.name, 'lambda/forward/op:0')
        var_1, op_1 = comp(['var', 'op'])
        self.assertEqual(var_1.name, 'lambda/var:0')
        self.assertEqual(op_1.name, 'lambda/forward_1/op:0')
        self.assertIs(var_1, var)
