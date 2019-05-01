import unittest

import pytest
from mock import Mock

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


class AnnealingScalarTestCase(unittest.TestCase):

    def test_freq_1(self):
        loop = Mock(steps=100)

        v = AnnealingScalar(loop, initial_value=1., ratio=.5, epochs=1)
        loop.epoch = 0
        self.assertEqual(v.get(), 1.)
        loop.epoch = 1
        self.assertEqual(v.get(), 1.)
        loop.epoch = 2
        self.assertEqual(v.get(), .5)
        loop.epoch = 3
        self.assertEqual(v.get(), .25)

        # test cache
        self.assertEqual(v.get(), .25)

    def test_freq_3(self):
        loop = Mock(epochs=100)

        v = AnnealingScalar(loop, initial_value=1., ratio=.5, steps=3)
        loop.step = 0
        self.assertEqual(v.get(), 1.)
        loop.step = 1
        self.assertEqual(v.get(), 1.)
        loop.step = 2
        self.assertEqual(v.get(), 1.)
        loop.step = 3
        self.assertEqual(v.get(), 1.)

        loop.step = 4
        self.assertEqual(v.get(), .5)
        loop.step = 5
        self.assertEqual(v.get(), .5)
        loop.step = 6
        self.assertEqual(v.get(), .5)

        loop.step = 7
        self.assertEqual(v.get(), .25)

        # test cache
        self.assertEqual(v.get(), .25)

    def test_min_value(self):
        loop = Mock(steps=100)

        v = AnnealingScalar(loop, initial_value=1., ratio=.5, epochs=1,
                            min_value=.1)
        loop.epoch = 0
        self.assertEqual(v.get(), 1.)
        loop.epoch = 1
        self.assertEqual(v.get(), 1.)
        loop.epoch = 2
        self.assertEqual(v.get(), .5)
        loop.epoch = 3
        self.assertEqual(v.get(), .25)
        loop.epoch = 4
        self.assertEqual(v.get(), .125)
        loop.epoch = 5
        self.assertEqual(v.get(), .1)

        # test cache
        self.assertEqual(v.get(), .1)

    def test_max_value(self):
        loop = Mock(epochs=100)

        v = AnnealingScalar(loop, initial_value=1., ratio=1.5, steps=1,
                            max_value=3.)
        loop.step = 0
        self.assertEqual(v.get(), 1.)
        loop.step = 1
        self.assertEqual(v.get(), 1.)
        loop.step = 2
        self.assertEqual(v.get(), 1.5)
        loop.step = 3
        self.assertEqual(v.get(), 2.25)
        loop.step = 4
        self.assertEqual(v.get(), 3.)

        # test cache
        self.assertEqual(v.get(), 3.)

    def test_errors(self):
        loop = Mock()

        with pytest.raises(ValueError,
                           match='`initial_value` must >= `min_value`: '
                                 'initial_value 1.0 vs min_value 2.0'):
            _ = AnnealingScalar(loop, initial_value=1., ratio=.5, min_value=2.)
        with pytest.raises(ValueError,
                           match='`initial_value` must <= `max_value`: '
                                 'initial_value 1.0 vs max_value 0.5'):
            _ = AnnealingScalar(loop, initial_value=1., ratio=.5, max_value=.5)
        with pytest.raises(ValueError,
                           match='`min_value` must <= `max_value`: '
                                 'min_value 1.0 vs max_value 0.5'):
            _ = AnnealingScalar(loop, initial_value=1., ratio=.5, min_value=1.0,
                                max_value=.5)
        with pytest.raises(ValueError,
                           match='One and only one of `epochs` and `steps` '
                                 'should be specified'):
            _ = AnnealingScalar(loop, initial_value=1., ratio=.5)
        with pytest.raises(ValueError,
                           match='One and only one of `epochs` and `steps` '
                                 'should be specified'):
            _ = AnnealingScalar(loop, initial_value=1., ratio=.5, epochs=1,
                                steps=1)
        with pytest.raises(ValueError,
                           match='`epochs` must be positive: -1'):
            _ = AnnealingScalar(loop, initial_value=1., ratio=.5, epochs=-1)
        with pytest.raises(ValueError,
                           match='`steps` must be positive: -1'):
            _ = AnnealingScalar(loop, initial_value=1., ratio=.5, steps=-1)
