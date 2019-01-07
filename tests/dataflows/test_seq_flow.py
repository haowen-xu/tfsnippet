import unittest

import numpy as np
import pytest

from tfsnippet.dataflows import DataFlow
from tfsnippet.dataflows.seq_flow import SeqFlow


class SeqFlowTestCase(unittest.TestCase):

    def test_seq(self):
        df = DataFlow.seq(
            1, 9, 2, batch_size=3, shuffle=False, skip_incomplete=False,
            dtype=np.int64
        )
        self.assertIsInstance(df, SeqFlow)
        self.assertEqual(1, df.array_count)
        self.assertEqual(4, df.data_length)
        self.assertEqual(((),), df.data_shapes)
        self.assertEqual(3, df.batch_size)
        self.assertFalse(df.is_shuffled)
        self.assertFalse(df.skip_incomplete)
        self.assertEqual(1, df.start)
        self.assertEqual(9, df.stop)
        self.assertEqual(2, df.step)

    def test_property(self):
        df = SeqFlow(
            1, 9, 2, batch_size=3, shuffle=True, skip_incomplete=True,
            dtype=np.int64
        )
        self.assertEqual(1, df.array_count)
        self.assertEqual(4, df.data_length)
        self.assertEqual(((),), df.data_shapes)
        self.assertEqual(3, df.batch_size)
        self.assertTrue(df.skip_incomplete)
        self.assertTrue(df.is_shuffled)
        self.assertEqual(1, df.start)
        self.assertEqual(9, df.stop)
        self.assertEqual(2, df.step)

        # test default options
        df = SeqFlow(1, 9, batch_size=3)
        self.assertFalse(df.skip_incomplete)
        self.assertFalse(df.is_shuffled)
        self.assertEqual(1, df.step)

    def test_errors(self):
        with pytest.raises(
                ValueError, match='`batch_size` is required'):
            _ = SeqFlow(1, 9, 2)

    def test_iterator(self):
        # test single array, without shuffle, no ignore
        b = [a[0] for a in SeqFlow(1, 9, 2, batch_size=3)]
        self.assertEqual(2, len(b))
        np.testing.assert_array_equal([1, 3, 5], b[0])
        np.testing.assert_array_equal([7], b[1])

        # test single array, without shuffle, ignore
        b = [a[0] for a in SeqFlow(1, 9, 2, batch_size=3, skip_incomplete=True)]
        self.assertEqual(1, len(b))
        np.testing.assert_array_equal([1, 3, 5], b[0])

        # test single array, with shuffle, no ignore
        b = [a[0] for a in SeqFlow(1, 9, 2, batch_size=3, shuffle=True)]
        self.assertEqual(2, len(b))
        np.testing.assert_array_equal(
            np.arange(1, 9, 2), sorted(np.concatenate(b)))
