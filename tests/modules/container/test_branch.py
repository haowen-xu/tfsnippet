import pytest
import six
import tensorflow as tf

from tfsnippet.modules import ListMapper, DictMapper, Lambda


class ListMapperTestCase(tf.test.TestCase):

    def test_outputs(self):
        net = ListMapper([
            Lambda(lambda x: x * tf.get_variable('x', initializer=0.)),
            (lambda x: x * tf.get_variable('y', initializer=0.)),
        ])
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        outputs = net(inputs)
        self.assertIsInstance(outputs, list)
        for v in outputs:
            self.assertIsInstance(v, tf.Tensor)

        _ = net(inputs)
        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['lambda/x:0',
             'list_mapper/_1/y:0']
        )

    def test_empty_mapper(self):
        with pytest.raises(ValueError, match='`mapper` must not be empty'):
            _ = ListMapper([])


class DictMapperTestCase(tf.test.TestCase):

    def test_outputs(self):
        net = DictMapper({
            'a': Lambda(lambda x: x * tf.get_variable('x', initializer=0.)),
            'b': lambda x: x * tf.get_variable('y', initializer=0.)
        })
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        output = net(inputs)
        self.assertIsInstance(output, dict)
        self.assertEqual(sorted(output.keys()), ['a', 'b'])
        for v in six.itervalues(output):
            self.assertIsInstance(v, tf.Tensor)

        _ = net(inputs)
        self.assertEqual(
            sorted(v.name for v in tf.global_variables()),
            ['dict_mapper/b/y:0',
             'lambda/x:0']
        )

    def test_empty_mapper(self):
        with pytest.raises(ValueError, match='`mapper` must not be empty'):
            _ = DictMapper({})

    def test_invalid_key(self):
        for k in ['.', '', '90ab', 'abc.def']:
            with pytest.raises(
                ValueError, match='The key for `DictMapper` must be a valid '
                                  'Python identifier'):
                _ = DictMapper({k: lambda x: x})
