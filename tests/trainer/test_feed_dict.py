import unittest

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


if __name__ == '__main__':
    unittest.main()
