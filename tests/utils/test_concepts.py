import unittest

import pytest
from mock import MagicMock

from tfsnippet.utils import NoReentrantContext, OneTimeContext


class _ContextA(NoReentrantContext):

    def __init__(self):
        self._enter = MagicMock(return_value=123)
        self._exit = MagicMock()


class _ContextB(OneTimeContext):

    def __init__(self):
        self._enter = MagicMock(return_value=456)
        self._exit = MagicMock()


class NoReentrantContextTestCase(unittest.TestCase):

    def text_context(self):
        ctx = _ContextA()

        self.assertFalse(ctx._is_entered)
        self.assertEquals(0, ctx._enter.call_count)
        self.assertEquals(0, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='_ContextA is not currently entered'):
            ctx._require_entered()

        with ctx as x:
            self.assertEqual(123, x)
            self.assertTrue(ctx._is_entered)
            self.assertEquals(1, ctx._enter.call_count)
            self.assertEquals(0, ctx._exit.call_count)
            _ = ctx._require_entered()

            with pytest.raises(
                    RuntimeError, match='_ContextA is not reentrant'):
                with ctx:
                    pass
            self.assertTrue(ctx._is_entered)
            self.assertEquals(1, ctx._enter.call_count)
            self.assertEquals(0, ctx._exit.call_count)

        self.assertFalse(ctx._is_entered)
        self.assertEquals(1, ctx._enter.call_count)
        self.assertEquals(1, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='_ContextA is not currently entered'):
            ctx._require_entered()

        with ctx as x:
            self.assertEqual(123, x)
            self.assertTrue(ctx._is_entered)
            self.assertEquals(2, ctx._enter.call_count)
            self.assertEquals(1, ctx._exit.call_count)
            _ = ctx._require_entered()

        self.assertFalse(ctx._is_entered)
        self.assertEquals(2, ctx._enter.call_count)
        self.assertEquals(2, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='_ContextA is not currently entered'):
            ctx._require_entered()


class OneTimeContextTestCase(unittest.TestCase):

    def text_context(self):
        ctx = _ContextB()

        self.assertFalse(ctx._is_entered)
        self.assertEquals(0, ctx._enter.call_count)
        self.assertEquals(0, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='_ContextB is not currently entered'):
            ctx._require_entered()

        with ctx as x:
            _ = ctx._require_entered()
            self.assertEqual(123, x)
            self.assertTrue(ctx._is_entered)
            self.assertEquals(1, ctx._enter.call_count)
            self.assertEquals(0, ctx._exit.call_count)

            with pytest.raises(
                    RuntimeError, match='_ContextB is not reentrant'):
                with ctx:
                    pass
            self.assertTrue(ctx._is_entered)
            self.assertEquals(1, ctx._enter.call_count)
            self.assertEquals(0, ctx._exit.call_count)

        self.assertFalse(ctx._is_entered)
        self.assertEquals(1, ctx._enter.call_count)
        self.assertEquals(1, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='_ContextB is not currently entered'):
            ctx._require_entered()

        with pytest.raises(
                RuntimeError, match='The one-time context _ContextB has '
                                    'already been entered, thus cannot be '
                                    'entered again'):
            with ctx:
                pass
