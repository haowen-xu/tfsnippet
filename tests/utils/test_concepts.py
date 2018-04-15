import unittest

import pytest
from mock import MagicMock

from tfsnippet.utils import OpenCloseContext


class MyContext(OpenCloseContext):

    def __init__(self):
        self._open = MagicMock()
        self._close = MagicMock()


class OpenCloseContextTestCase(unittest.TestCase):

    def test_open(self):
        ctx = MyContext()
        self.assertFalse(ctx._has_opened)
        self.assertEquals(0, ctx._open.call_count)

        ctx.open()
        self.assertTrue(ctx._has_opened)
        self.assertEquals(1, ctx._open.call_count)

        with pytest.raises(
                RuntimeError, match='The MyContext has been opened'):
            ctx.open()

    def test_close(self):
        ctx = MyContext()
        self.assertFalse(ctx._has_closed)
        self.assertEquals(0, ctx._close.call_count)

        ctx.close()  # should have no side-effect
        self.assertFalse(ctx._has_closed)
        self.assertEquals(0, ctx._close.call_count)

        ctx.open()
        ctx.close()
        self.assertTrue(ctx._has_closed)
        self.assertEquals(1, ctx._close.call_count)

        ctx.close()  # should have no side-effect
        self.assertEquals(1, ctx._close.call_count)

    def test_require_alive(self):
        ctx = MyContext()
        with pytest.raises(
                RuntimeError, match='The MyContext has not been opened'):
            ctx._require_alive()
        ctx.open()
        ctx._require_alive()
        ctx.close()
        with pytest.raises(
                RuntimeError, match='The MyContext has been closed'):
            ctx._require_alive()

    def test_enter_exit(self):
        ctx = MyContext()
        self.assertFalse(ctx._has_opened)
        self.assertFalse(ctx._has_closed)
        self.assertEquals(0, ctx._open.call_count)
        self.assertEquals(0, ctx._close.call_count)

        with ctx:
            self.assertTrue(ctx._has_opened)
            self.assertFalse(ctx._has_closed)
            self.assertEquals(1, ctx._open.call_count)
            self.assertEquals(0, ctx._close.call_count)

        self.assertTrue(ctx._has_opened)
        self.assertTrue(ctx._has_closed)
        self.assertEquals(1, ctx._open.call_count)
        self.assertEquals(1, ctx._close.call_count)
