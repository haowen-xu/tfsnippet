import unittest

import numpy as np
import pytest

from tfsnippet.dataflow import SeqFlow, DataFlow


class SeqFlowTestCase(unittest.TestCase):

    def test_seq(self):
        df = DataFlow.seq(
            1, 9, 2, batch_size=3, shuffle=False, skip_incomplete=False,
            dtype=np.int64
        )
        self.assertIsInstance(df, SeqFlow)
        self.assertEquals(1, df.array_count)
        self.assertEquals(4, df.data_length)
        self.assertEquals(((),), df.data_shapes)
        self.assertEquals(3, df.batch_size)
        self.assertFalse(df.is_shuffled)
        self.assertFalse(df.skip_incomplete)
        self.assertEquals(1, df.start)
        self.assertEquals(9, df.stop)
        self.assertEquals(2, df.step)

    def test_property(self):
        df = SeqFlow(
            1, 9, 2, batch_size=3, shuffle=True, skip_incomplete=True,
            dtype=np.int64
        )
        self.assertEquals(1, df.array_count)
        self.assertEquals(4, df.data_length)
        self.assertEquals(((),), df.data_shapes)
        self.assertEquals(3, df.batch_size)
        self.assertTrue(df.skip_incomplete)
        self.assertTrue(df.is_shuffled)
        self.assertEquals(1, df.start)
        self.assertEquals(9, df.stop)
        self.assertEquals(2, df.step)

        # test default options
        df = SeqFlow(1, 9, batch_size=3)
        self.assertFalse(df.skip_incomplete)
        self.assertFalse(df.is_shuffled)
        self.assertEquals(1, df.step)

    def test_errors(self):
        with pytest.raises(
                ValueError, match='`batch_size` is required'):
            _ = SeqFlow(1, 9, 2)

    def test_iterator(self):
        # test single array, without shuffle, no ignore
        b = [a[0] for a in SeqFlow(1, 9, 2, batch_size=3)]
        self.assertEquals(2, len(b))
        np.testing.assert_array_equal([1, 3, 5], b[0])
        np.testing.assert_array_equal([7], b[1])

        # test single array, without shuffle, ignore
        b = [a[0] for a in SeqFlow(1, 9, 2, batch_size=3, skip_incomplete=True)]
        self.assertEquals(1, len(b))
        np.testing.assert_array_equal([1, 3, 5], b[0])

        # test single array, with shuffle, no ignore
        b = [a[0] for a in SeqFlow(1, 9, 2, batch_size=3, shuffle=True)]
        self.assertEquals(2, len(b))
        np.testing.assert_array_equal(
            np.arange(1, 9, 2), sorted(np.concatenate(b)))


if __name__ == '__main__':
    unittest.main()
