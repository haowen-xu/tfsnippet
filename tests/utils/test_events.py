import unittest

import pytest
from mock import Mock

from tfsnippet.utils import EventSource


class EventSourceTestCase(unittest.TestCase):

    def test_events(self):
        f1 = Mock()
        f2 = Mock()

        events = EventSource()
        events.on('ev1', f1)
        events.on('ev2', f1)
        events.on('ev2', f2)

        events.fire('ev1', 123, value=456)
        self.assertEqual(f1.call_args, ((123,), {'value': 456}))
        self.assertFalse(f2.called)

        f1.reset_mock()
        f2.reset_mock()
        events.fire('ev2', 7788, v=9988)
        self.assertEqual(f1.call_args, ((7788,), {'v': 9988}))
        self.assertEqual(f2.call_args, ((7788,), {'v': 9988}))

        f1.reset_mock()
        f2.reset_mock()
        events.off('ev2', f1)
        events.fire('ev2', 999, k=888)
        self.assertFalse(f1.called)
        self.assertEqual(f2.call_args, ((999,), {'k': 888}))

        events.fire('ev1', 321, value=654)
        self.assertEqual(f1.call_args, ((321,), {'value': 654}))

        events = EventSource(['ev1'])
        events.on('ev1', f1)

        f1.reset_mock()
        events.fire('ev1', 123, value=456)
        self.assertEqual(f1.call_args, ((123,), {'value': 456}))

    def test_order(self):
        dest = []

        def f(x):
            dest.append(x)

        events = EventSource()
        events.on('ev', lambda: f(1))
        events.on('ev', lambda: f(2))
        events.on('ev', lambda: f(3))

        events.fire('ev')
        self.assertListEqual(dest, [1, 2, 3])

        del dest[:]
        events.reverse_fire('ev')
        self.assertListEqual(dest, [3, 2, 1])

    def test_clear(self):
        f1 = Mock()
        f2 = Mock()

        events = EventSource()
        events.on('ev1', f1)
        events.on('ev1', f2)
        events.on('ev2', f1)
        events.on('ev2', f2)
        events.on('ev3', f1)
        events.on('ev3', f2)

        events.fire('ev1')
        events.fire('ev2')
        events.fire('ev3')
        self.assertEqual(f1.call_count, 3)
        self.assertEqual(f2.call_count, 3)

        f1.reset_mock()
        f2.reset_mock()
        events.clear_event_handlers('ev1')
        events.clear_event_handlers('ev4')
        events.fire('ev1')
        events.fire('ev2')
        events.fire('ev3')
        self.assertEqual(f1.call_count, 2)
        self.assertEqual(f2.call_count, 2)

        f1.reset_mock()
        f2.reset_mock()
        events.clear_event_handlers()
        events.fire('ev1')
        events.fire('ev2')
        events.fire('ev3')
        self.assertEqual(f1.call_count, 0)
        self.assertEqual(f2.call_count, 0)

    def test_errors(self):
        f = Mock()
        events = EventSource(['ev1'])
        with pytest.raises(KeyError, match='`event_key` is not allowed'):
            events.on('ev2', f)
        with pytest.raises(KeyError, match='`event_key` is not allowed'):
            events.fire('ev2', 123, value=456)
        with pytest.raises(ValueError, match='`handler` is not a registered '
                                             'event handler of event `ev1`'):
            events.off('ev1', f)
