import unittest

import pytest
from mock import MagicMock

from tfsnippet.dataflow import DataFlow


class _DataFlow(DataFlow):

    def __init__(self):
        self._minibatch_iterator = MagicMock(return_value=[123])


class DataFlowTestCase(unittest.TestCase):

    def test_iter(self):
        df = _DataFlow()

        self.assertFalse(df._is_iter_entered)
        self.assertEquals(0, df._minibatch_iterator.call_count)

        for x in df:
            self.assertEquals(123, x)
            self.assertTrue(df._is_iter_entered)
            self.assertEquals(1, df._minibatch_iterator.call_count)

            with pytest.raises(
                    RuntimeError, match='_DataFlow.__iter__ is not reentrant'):
                for _ in df:
                    pass

        self.assertFalse(df._is_iter_entered)
        self.assertEquals(1, df._minibatch_iterator.call_count)


if __name__ == '__main__':
    unittest.main()
