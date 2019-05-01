import unittest

import six
import tensorflow as tf

from tfsnippet.trainer import *
from tfsnippet.scaffold import ScheduledVariable
from tfsnippet.utils import ensure_variables_initialized


class MyDynamicValue(DynamicValue):

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class ResolveFeedDictTestCase(tf.test.TestCase):

    def test_copy(self):
        with self.test_session():
            d = {
                'a': 12,
                'b': ScheduledVariable('b', 34),
                'c': MyDynamicValue(56),
                'd': lambda: 78,
            }
            ensure_variables_initialized()
            d2 = resolve_feed_dict(d)
            self.assertIsNot(d2, d)
            self.assertDictEqual({'a': 12, 'b': 34, 'c': 56, 'd': 78}, d2)
            self.assertIsInstance(d['b'], ScheduledVariable)
            self.assertIsInstance(d['c'], MyDynamicValue)

    def test_inplace(self):
        with self.test_session():
            d = {
                'a': 12,
                'b': ScheduledVariable('b', 34),
                'c': MyDynamicValue(56),
                'd': lambda: 78,
            }
            ensure_variables_initialized()
            self.assertIs(d, resolve_feed_dict(d, inplace=True))
            self.assertDictEqual({'a': 12, 'b': 34, 'c': 56, 'd': 78}, d)


class MergeFeedDictTestCase(unittest.TestCase):

    def test_merge(self):
        self.assertDictEqual(
            {'a': 10, 'b': 200, 'c': 300, 'd': 4},
            merge_feed_dict(
                None,
                {'a': 1, 'b': 2, 'c': 3, 'd': 4},
                iter([('a', 10), ('b', 20), ('c', 30)]),
                None,
                six.iteritems({'b': 200, 'c': 300})
            )
        )
