import unittest

import pytest
from mock import Mock

from tfsnippet.trainer import *


class HookPriorityTestCase(unittest.TestCase):

    def test_priority(self):
        self.assertLess(HookPriority.VALIDATION, HookPriority.DEFAULT)
        self.assertLess(HookPriority.DEFAULT, HookPriority.ANNEALING)
        self.assertLess(HookPriority.ANNEALING, HookPriority.LOGGING)


class HookEntryTestCase(unittest.TestCase):

    def test_props(self):
        callback = Mock(return_value=None)
        e = HookEntry(callback, 2, 300, 999)
        self.assertEquals(callback, e.callback)
        self.assertEquals(2, e.freq)
        self.assertEquals(300, e.priority)
        self.assertEquals(2, e.counter)
        self.assertEquals(999, e.birth)

    def test_maybe_call(self):
        callback = Mock(return_value=None)
        e = HookEntry(callback, 2, 300, 999)
        e.maybe_call()
        self.assertEquals(0, callback.call_count)
        self.assertEquals(1, e.counter)
        e.maybe_call()
        self.assertEquals(1, callback.call_count)
        self.assertEquals(2, e.counter)
        e.maybe_call()
        self.assertEquals(1, callback.call_count)
        e.maybe_call()
        self.assertEquals(2, callback.call_count)

    def test_maybe_call_error(self):
        def throw_error():
            raise RuntimeError('callback error')

        e = HookEntry(throw_error, 2, 300, 999)
        e.maybe_call()
        with pytest.raises(RuntimeError, match='callback error'):
            e.maybe_call()
        e.maybe_call()
        with pytest.raises(RuntimeError, match='callback error'):
            e.maybe_call()

    def test_reset_counter(self):
        e = HookEntry(Mock(return_value=None), 2, 300, 999)
        e.maybe_call()
        self.assertEquals(1, e.counter)
        e.reset_counter()
        self.assertEquals(2, e.counter)

    def test_sort_key(self):
        e = HookEntry(Mock(return_value=None), freq=2, priority=300, birth=999)
        self.assertEquals((300, 999), e.sort_key())

    def test_one_freq(self):
        callback = Mock(return_value=None)
        e = HookEntry(callback, freq=1, priority=100, birth=1)
        e.maybe_call()
        e.maybe_call()
        self.assertEquals(2, callback.call_count)

    def test_fractional_freq(self):
        callback = Mock(return_value=None)
        e = HookEntry(callback, freq=.9, priority=100, birth=1)
        e.maybe_call()
        e.maybe_call()
        self.assertEquals(2, callback.call_count)


if __name__ == '__main__':
    unittest.main()
