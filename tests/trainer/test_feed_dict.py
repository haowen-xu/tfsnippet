import unittest

import six

from tfsnippet.trainer import *


class ResolveFeedDictTestCase(unittest.TestCase):

    def test_copy(self):
        d = {
            'a': 12,
            'b': SimpleDynamicValue(34),
            'c': SimpleDynamicValue(SimpleDynamicValue(56)),
            'd': lambda: 78,
        }
        d2 = resolve_feed_dict(d)
        self.assertIsNot(d2, d)
        self.assertDictEqual({'a': 12, 'b': 34, 'c': 56, 'd': 78}, d2)
        self.assertIsInstance(d['b'], DynamicValue)
        self.assertIsInstance(d['c'], DynamicValue)

    def test_inplace(self):
        d = {
            'a': 12,
            'b': SimpleDynamicValue(34),
            'c': SimpleDynamicValue(SimpleDynamicValue(56)),
            'd': lambda: 78,
        }
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
