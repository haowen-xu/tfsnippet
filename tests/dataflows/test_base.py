import unittest

import numpy as np
import pytest
from mock import MagicMock

from tfsnippet.dataflows import DataFlow
from tfsnippet.dataflows.array_flow import ArrayFlow


class _DataFlow(DataFlow):

    def __init__(self):
        self._minibatch_iterator = MagicMock(return_value=[123])


class DataFlowTestCase(unittest.TestCase):

    def test_iter(self):
        df = _DataFlow()

        self.assertFalse(df._is_iter_entered)
        self.assertEqual(0, df._minibatch_iterator.call_count)

        for x in df:
            self.assertEqual(123, x)
            self.assertTrue(df._is_iter_entered)
            self.assertEqual(1, df._minibatch_iterator.call_count)

            with pytest.raises(
                    RuntimeError, match='_DataFlow.__iter__ is not reentrant'):
                for _ in df:
                    pass

        self.assertFalse(df._is_iter_entered)
        self.assertEqual(1, df._minibatch_iterator.call_count)

    def test_get_arrays(self):
        with pytest.raises(ValueError, match='empty, cannot convert to arrays'):
            _ = DataFlow.arrays([np.arange(0)], batch_size=5).get_arrays()

        # test one batch
        df = DataFlow.arrays([np.arange(5), np.arange(5, 10)], batch_size=6)
        arrays = df.get_arrays()
        np.testing.assert_equal(np.arange(5), arrays[0])
        np.testing.assert_equal(np.arange(5, 10), arrays[1])

        # test two batches
        df = DataFlow.arrays([np.arange(10), np.arange(10, 20)], batch_size=6)
        arrays = df.get_arrays()
        np.testing.assert_equal(np.arange(10), arrays[0])
        np.testing.assert_equal(np.arange(10, 20), arrays[1])

        # test to_arrays_flow
        df2 = df.to_arrays_flow(batch_size=6)
        self.assertIsInstance(df2, ArrayFlow)

    def test_implicit_iterator(self):
        df = DataFlow.arrays([np.arange(3)], batch_size=2)
        self.assertIsNone(df.current_batch)

        np.testing.assert_equal([[0, 1]], df.next_batch())
        np.testing.assert_equal([[0, 1]], df.current_batch)
        np.testing.assert_equal([[2]], df.next_batch())
        np.testing.assert_equal([[2]], df.current_batch)
        with pytest.raises(StopIteration):
            _ = df.next_batch()
        self.assertIsNone(df.current_batch)

        np.testing.assert_equal([[0, 1]], df.next_batch())
        np.testing.assert_equal([[0, 1]], df.current_batch)
