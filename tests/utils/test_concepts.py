import unittest

import pytest
from mock import MagicMock, Mock

from tfsnippet.utils import (AutoInitAndCloseable, Disposable,
                             NoReentrantContext, DisposableContext)


class AutoInitAndCloseableTestCase(unittest.TestCase):

    def test_init_close(self):
        lazy_init = AutoInitAndCloseable()
        lazy_init._init = Mock()
        lazy_init._close = Mock()
        self.assertEqual(0, lazy_init._init.call_count)
        self.assertEqual(0, lazy_init._close.call_count)

        lazy_init.init()
        self.assertEqual(1, lazy_init._init.call_count)
        self.assertEqual(0, lazy_init._close.call_count)

        lazy_init.init()
        self.assertEqual(1, lazy_init._init.call_count)
        self.assertEqual(0, lazy_init._close.call_count)

        lazy_init.close()
        self.assertEqual(1, lazy_init._init.call_count)
        self.assertEqual(1, lazy_init._close.call_count)

        lazy_init.close()
        self.assertEqual(1, lazy_init._init.call_count)
        self.assertEqual(1, lazy_init._close.call_count)

        lazy_init.init()
        self.assertEqual(2, lazy_init._init.call_count)
        self.assertEqual(1, lazy_init._close.call_count)

    def test_context(self):
        lazy_init = AutoInitAndCloseable()
        lazy_init._init = Mock()
        lazy_init._close = Mock()
        self.assertEqual(0, lazy_init._init.call_count)
        self.assertEqual(0, lazy_init._close.call_count)

        with lazy_init:
            self.assertEqual(1, lazy_init._init.call_count)
            self.assertEqual(0, lazy_init._close.call_count)
            with lazy_init:
                self.assertEqual(1, lazy_init._init.call_count)
                self.assertEqual(0, lazy_init._close.call_count)
            self.assertEqual(1, lazy_init._init.call_count)
            self.assertEqual(1, lazy_init._close.call_count)
        self.assertEqual(1, lazy_init._init.call_count)
        self.assertEqual(1, lazy_init._close.call_count)

        with lazy_init:
            self.assertEqual(2, lazy_init._init.call_count)
            self.assertEqual(1, lazy_init._close.call_count)
        self.assertEqual(2, lazy_init._init.call_count)
        self.assertEqual(2, lazy_init._close.call_count)


class DisposableTestCase(unittest.TestCase):

    def test_everything(self):
        disposable = Disposable()
        disposable._check_usage_and_set_used()
        with pytest.raises(
                RuntimeError, match='Disposable object cannot be used twice'):
            disposable._check_usage_and_set_used()


class _ContextA(NoReentrantContext):

    def __init__(self):
        self._enter = MagicMock(return_value=123)
        self._exit = MagicMock()


class _ContextB(DisposableContext):

    def __init__(self):
        self._enter = MagicMock(return_value=456)
        self._exit = MagicMock()


class NoReentrantContextTestCase(unittest.TestCase):

    def test_everything(self):
        ctx = _ContextA()

        self.assertFalse(ctx._is_entered)
        self.assertEqual(0, ctx._enter.call_count)
        self.assertEqual(0, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='Context is required be entered'):
            ctx._require_entered()

        with ctx as x:
            self.assertEqual(123, x)
            self.assertTrue(ctx._is_entered)
            self.assertEqual(1, ctx._enter.call_count)
            self.assertEqual(0, ctx._exit.call_count)
            _ = ctx._require_entered()

            with pytest.raises(
                    RuntimeError, match='Context is not reentrant'):
                with ctx:
                    pass
            self.assertTrue(ctx._is_entered)
            self.assertEqual(1, ctx._enter.call_count)
            self.assertEqual(0, ctx._exit.call_count)

        self.assertFalse(ctx._is_entered)
        self.assertEqual(1, ctx._enter.call_count)
        self.assertEqual(1, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='Context is required be entered'):
            ctx._require_entered()

        with ctx as x:
            self.assertEqual(123, x)
            self.assertTrue(ctx._is_entered)
            self.assertEqual(2, ctx._enter.call_count)
            self.assertEqual(1, ctx._exit.call_count)
            _ = ctx._require_entered()

        self.assertFalse(ctx._is_entered)
        self.assertEqual(2, ctx._enter.call_count)
        self.assertEqual(2, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='Context is required be entered'):
            ctx._require_entered()


class DisposableContextTestCase(unittest.TestCase):

    def test_everything(self):
        ctx = _ContextB()

        self.assertFalse(ctx._is_entered)
        self.assertEqual(0, ctx._enter.call_count)
        self.assertEqual(0, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='Context is required be entered'):
            ctx._require_entered()

        with ctx as x:
            _ = ctx._require_entered()
            self.assertEqual(123, x)
            self.assertTrue(ctx._is_entered)
            self.assertEqual(1, ctx._enter.call_count)
            self.assertEqual(0, ctx._exit.call_count)

            with pytest.raises(
                    RuntimeError, match='Context is not reentrant'):
                with ctx:
                    pass
            self.assertTrue(ctx._is_entered)
            self.assertEqual(1, ctx._enter.call_count)
            self.assertEqual(0, ctx._exit.call_count)

        self.assertFalse(ctx._is_entered)
        self.assertEqual(1, ctx._enter.call_count)
        self.assertEqual(1, ctx._exit.call_count)
        with pytest.raises(
                RuntimeError, match='Context is required be entered'):
            ctx._require_entered()

        with pytest.raises(
                RuntimeError,
                match='A disposable context cannot be entered twice'):
            with ctx:
                pass
