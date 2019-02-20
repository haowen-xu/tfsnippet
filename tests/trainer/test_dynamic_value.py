import unittest

from tfsnippet.trainer import *


class DynamicValuesTestCase(unittest.TestCase):

    def test_DynamicValue(self):
        x = [1]

        class MyValue(DynamicValue):
            def get(self):
                return x[0]

        v = MyValue()
        self.assertEqual(v.get(), 1)
        x[0] = 123
        self.assertEqual(v.get(), 123)
