import pytest
import tensorflow as tf

from tfsnippet.modules import Module, Sequential


class SequentialTestCase(tf.test.TestCase):

    def test_scopes(self):
        def f(inputs):
            return tf.get_variable('v', shape=()) * inputs

        class C(Module):

            def __init__(self, flag):
                self.flag = flag
                super(C, self).__init__()

            def _forward(self, inputs, **kwargs):
                v = tf.get_variable('a' if self.flag else 'b', shape=())
                return v * inputs

        c = C(flag=True)
        seq = Sequential([
            f,
            c,
            tf.nn.relu,
            f,
            C(flag=False),
            c,
        ])
        self.assertEqual(tf.global_variables(), [])
        seq(tf.constant(1.))
        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['c/a:0',
             'c_1/b:0',
             'sequential/_0/v:0',
             'sequential/_3/v:0']
        )
        seq(tf.constant(2.))
        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['c/a:0',
             'c_1/b:0',
             'sequential/_0/v:0',
             'sequential/_3/v:0']
        )

    def test_errors(self):
        with pytest.raises(
                ValueError, message='`components` must not be empty'):
            Sequential([])
