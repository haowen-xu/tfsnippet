import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.layers import BaseLayer


class MyLayer(BaseLayer):

    def __init__(self, output, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self._build = Mock(wraps=self._build)
        self._apply = Mock(return_value=output)

    def _build(self, input=None):
        assert(tf.get_variable_scope().name == self.variable_scope.name)


class BaseLayerTestCase(tf.test.TestCase):

    def test_skeleton(self):
        input = tf.constant(123.)
        inputs = [tf.constant(12.), tf.constant(3.)]
        output = tf.constant(456.)

        # test call build manually
        layer = MyLayer(output)
        self.assertFalse(layer._has_built)
        layer.build()
        self.assertTrue(layer._has_built)
        self.assertEqual(layer._build.call_args, ((None,), {}))
        self.assertFalse(layer._apply.called)
        self.assertIs(layer.apply(input), output)
        self.assertEqual(layer._build.call_count, 1)
        self.assertEqual(layer._apply.call_args, ((input,), {}))

        with pytest.raises(RuntimeError, match='Layer has already been built'):
            _ = layer.build()

        layer = MyLayer(output)
        layer._build_require_input = True
        with pytest.raises(ValueError,
                           match='`MyLayer` requires `input` to build'):
            _ = layer.build()

        # test call build automatically
        layer = MyLayer(output)
        self.assertFalse(layer._has_built)
        self.assertIs(layer(inputs), output)
        self.assertEqual(layer._build.call_args, ((inputs,), {}))
        self.assertEqual(layer._apply.call_args, ((inputs,), {}))
