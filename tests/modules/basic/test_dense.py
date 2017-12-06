import tensorflow as tf

from tfsnippet.modules import Dense, Linear


class DenseTestCase(tf.test.TestCase):

    def test_variable_scopes(self):
        dense = Dense(100)
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        _ = dense(inputs)
        _ = dense(inputs)

        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['dense/fully_connected/biases:0',
             'dense/fully_connected/weights:0']
        )


class LinearTestCase(tf.test.TestCase):

    def test_variable_scopes(self):
        linear = Linear(100)
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        _ = linear(inputs)
        _ = linear(inputs)

        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['linear/fully_connected/biases:0',
             'linear/fully_connected/weights:0']
        )
