import unittest

import pytest

from tfsnippet.trainer import *


class DynamicValuesTestCase(unittest.TestCase):

    def test_SimpleDynamicValue(self):
        v = SimpleDynamicValue(123)
        self.assertEqual(123, v.get())
        v.set(456)
        self.assertEqual(456, v.get())
        v.set(SimpleDynamicValue(789))
        self.assertEqual(789, v.get())

        with pytest.raises(ValueError, match='Cannot set the value to `self`.'):
            v.set(v)

    def test_AnnealingDynamicValue(self):
        # test without min_value
        v = AnnealingDynamicValue(1, .5)
        self.assertEqual(1, v.get())
        self.assertEqual(.5, v.ratio)
        v.ratio = .25
        self.assertEqual(.25, v.ratio)

        v.anneal()
        self.assertEqual(.25, v.get())
        v.anneal()
        self.assertEqual(.0625, v.get())

        v.set(2.)
        self.assertEqual(2., v.get())
        v.anneal()
        self.assertEqual(.5, v.get())

        # test with min_value
        v = AnnealingDynamicValue(1, .5, 2)
        self.assertEqual(2, v.get())

        v = AnnealingDynamicValue(1, .5, .5)
        self.assertEqual(1, v.get())
        v.anneal()
        self.assertEqual(.5, v.get())
        v.anneal()
        self.assertEqual(.5, v.get())


if __name__ == '__main__':
    unittest.main()
