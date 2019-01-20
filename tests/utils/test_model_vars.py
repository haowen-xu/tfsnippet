import tensorflow as tf

from tests.helper import assert_variables
from tfsnippet import model_variable, get_model_variables


class ModelVariableTestCase(tf.test.TestCase):

    def test_model_variable(self):
        a = model_variable('a', shape=(), dtype=tf.float32)
        b = model_variable('b', shape=(), dtype=tf.float32, trainable=False,
                           collections=['my_collection'])
        c = tf.get_variable('c', shape=(), dtype=tf.float32)

        assert_variables(['a'], trainable=True,
                         collections=[tf.GraphKeys.MODEL_VARIABLES])
        assert_variables(['b'], trainable=False,
                         collections=[tf.GraphKeys.MODEL_VARIABLES,
                                      'my_collection'])
        self.assertEqual(get_model_variables(), [a, b])
