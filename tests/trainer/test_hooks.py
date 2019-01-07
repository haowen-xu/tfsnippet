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
        self.assertEqual(callback, e.callback)
        self.assertEqual(2, e.freq)
        self.assertEqual(300, e.priority)
        self.assertEqual(2, e.counter)
        self.assertEqual(999, e.birth)

    def test_maybe_call(self):
        callback = Mock(return_value=None)
        e = HookEntry(callback, 2, 300, 999)
        e.maybe_call()
        self.assertEqual(0, callback.call_count)
        self.assertEqual(1, e.counter)
        e.maybe_call()
        self.assertEqual(1, callback.call_count)
        self.assertEqual(2, e.counter)
        e.maybe_call()
        self.assertEqual(1, callback.call_count)
        e.maybe_call()
        self.assertEqual(2, callback.call_count)

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
        self.assertEqual(1, e.counter)
        e.reset_counter()
        self.assertEqual(2, e.counter)

    def test_sort_key(self):
        e = HookEntry(Mock(return_value=None), freq=2, priority=300, birth=999)
        self.assertEqual((300, 999), e.sort_key())

    def test_one_freq(self):
        callback = Mock(return_value=None)
        e = HookEntry(callback, freq=1, priority=100, birth=1)
        e.maybe_call()
        e.maybe_call()
        self.assertEqual(2, callback.call_count)

    def test_fractional_freq(self):
        callback = Mock(return_value=None)
        e = HookEntry(callback, freq=.9, priority=100, birth=1)
        e.maybe_call()
        e.maybe_call()
        self.assertEqual(2, callback.call_count)

        callback = Mock(return_value=None)
        e = HookEntry(callback, freq=1.2, priority=100, birth=1)
        e.maybe_call()
        e.maybe_call()
        self.assertEqual(2, callback.call_count)


class HookListTestCase(unittest.TestCase):

    def test_add_and_call(self):
        f1 = Mock(return_value=None, __repr__=lambda o: 'f1')
        f2 = Mock(return_value=None, __repr__=lambda o: 'f2')
        f3 = Mock(return_value=None, __repr__=lambda o: 'f3')

        # test empty hook list
        hook_list = HookList()
        self.assertEqual('HookList()', repr(hook_list))
        hook_list.call_hooks()
        self.assertEqual(0, f1.call_count)
        self.assertEqual(0, f2.call_count)
        self.assertEqual(0, f3.call_count)

        # test hook list
        hook_list.add_hook(f1, freq=1, priority=HookPriority.DEFAULT)
        hook_list.add_hook(f2, freq=2, priority=HookPriority.VALIDATION)
        hook_list.add_hook(f3, freq=3, priority=HookPriority.DEFAULT)
        self.assertEqual(
            'HookList(f2:2,f1:1,f3:3)', repr(hook_list))

        hook_list.call_hooks()
        self.assertEqual(1, f1.call_count)
        self.assertEqual(0, f2.call_count)
        self.assertEqual(0, f3.call_count)

        hook_list.call_hooks()
        self.assertEqual(2, f1.call_count)
        self.assertEqual(1, f2.call_count)
        self.assertEqual(0, f3.call_count)

        hook_list.call_hooks()
        self.assertEqual(3, f1.call_count)
        self.assertEqual(1, f2.call_count)
        self.assertEqual(1, f3.call_count)

        hook_list.call_hooks()
        self.assertEqual(4, f1.call_count)
        self.assertEqual(2, f2.call_count)
        self.assertEqual(1, f3.call_count)

        # test reset counter
        hook_list.call_hooks()
        hook_list.reset()
        hook_list.call_hooks()
        hook_list.call_hooks()
        self.assertEqual(7, f1.call_count)
        self.assertEqual(3, f2.call_count)
        self.assertEqual(1, f3.call_count)

    def test_remove(self):
        hook_list = HookList()
        for i in range(5):
            hook_list.add_hook(Mock(return_value=None, index=i))
        hook_list.add_hook(Mock(return_value=None, index=5), priority=123456)
        self.assertEqual(
            [0, 1, 2, 3, 4, 5], [h.callback.index for h in hook_list._hooks])

        # remove by priority
        self.assertEqual(1, hook_list.remove_by_priority(123456))
        self.assertEqual(
            [0, 1, 2, 3, 4], [h.callback.index for h in hook_list._hooks])

        # remove a particular hook
        self.assertEqual(1, hook_list.remove(hook_list._hooks[-1].callback))
        self.assertEqual(
            [0, 1, 2, 3], [h.callback.index for h in hook_list._hooks])

        # remove even hooks
        self.assertEqual(
            2, hook_list.remove_if(lambda c, f, t: c.index % 2 == 0))
        self.assertEqual([1, 3], [h.callback.index for h in hook_list._hooks])

        # remove even hooks, with no hook removed
        self.assertEqual(
            0, hook_list.remove_if(lambda c, f, t: c.index % 2 == 0))
        self.assertEqual([1, 3], [h.callback.index for h in hook_list._hooks])

        # remove all hooks
        self.assertEqual(2, hook_list.remove_all())
        self.assertEqual([], [h.callback.index for h in hook_list._hooks])
        self.assertEqual(0, hook_list.remove_all())

    def test_error_add(self):
        with pytest.raises(ValueError, match='`freq` must be at least 1'):
            HookList().add_hook(lambda: None, freq=0)
