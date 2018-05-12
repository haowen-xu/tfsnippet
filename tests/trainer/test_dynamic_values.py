import unittest

import pytest

from tfsnippet.trainer import *


class DynamicValuesTestCase(unittest.TestCase):

    def test_SimpleDynamicValue(self):
        v = SimpleDynamicValue(123)
        self.assertEquals(123, v.get())
        v.set(456)
        self.assertEquals(456, v.get())
        v.set(SimpleDynamicValue(789))
        self.assertEquals(789, v.get())

        with pytest.raises(ValueError, match='Cannot set the value to `self`.'):
            v.set(v)

    def test_AnnealingDynamicValue(self):
        v = AnnealingDynamicValue(1, .5)
        self.assertEquals(1, v.get())
        self.assertEquals(.5, v.ratio)
        v.ratio = .25
        self.assertEquals(.25, v.ratio)

        v.anneal()
        self.assertEquals(.25, v.get())
        v.anneal()
        self.assertEquals(.0625, v.get())

        v.set(2.)
        self.assertEquals(2., v.get())
        v.anneal()
        self.assertEquals(.5, v.get())


if __name__ == '__main__':
    unittest.main()
