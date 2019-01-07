import unittest

import numpy as np
import pytest

from tfsnippet.dataflows import DataFlow
from tfsnippet.dataflows.array_flow import ArrayFlow


class ArrayFlowTestCase(unittest.TestCase):

    def test_arrays(self):
        arrays = [np.arange(5), np.arange(10).reshape([5, 2])]
        df = DataFlow.arrays(
            arrays, 4, shuffle=False, skip_incomplete=False)
        self.assertIsInstance(df, ArrayFlow)
        for i, arr in enumerate(arrays):
            self.assertIs(arr, df.the_arrays[i])
        self.assertEqual(2, df.array_count)
        self.assertEqual(5, df.data_length)
        self.assertEqual(((), (2,)), df.data_shapes)
        self.assertFalse(df.is_shuffled)
        self.assertFalse(df.skip_incomplete)

    def test_property(self):
        df = ArrayFlow(
            arrays=[np.arange(12).reshape([4, 3]), np.arange(4)],
            batch_size=5,
            shuffle=True,
            skip_incomplete=True
        )
        self.assertEqual(2, df.array_count)
        self.assertEqual(4, df.data_length)
        self.assertEqual(((3,), ()), df.data_shapes)
        self.assertEqual(5, df.batch_size)
        self.assertTrue(df.skip_incomplete)
        self.assertTrue(df.is_shuffled)

        # test default options
        df = ArrayFlow([np.arange(12)], 5)
        self.assertFalse(df.skip_incomplete)
        self.assertFalse(df.is_shuffled)

    def test_errors(self):
        with pytest.raises(
                ValueError, match='`arrays` must not be empty'):
            _ = ArrayFlow([], 3)
        with pytest.raises(
                ValueError, match='`arrays` must be numpy-like arrays'):
            _ = ArrayFlow([np.arange(3).tolist()], 3)
        with pytest.raises(
                ValueError, match='`arrays` must be at least 1-d arrays'):
            _ = ArrayFlow([np.array(0)], 3)
        with pytest.raises(
                ValueError, match='`arrays` must have the same data length'):
            _ = ArrayFlow([np.arange(3), np.arange(4)], 3)

    def test_iterator(self):
        # test single array, without shuffle, no ignore
        b = [a[0] for a in ArrayFlow([np.arange(12)], 5)]
        self.assertEqual(3, len(b))
        np.testing.assert_array_equal(np.arange(0, 5), b[0])
        np.testing.assert_array_equal(np.arange(5, 10), b[1])
        np.testing.assert_array_equal(np.arange(10, 12), b[2])

        # test single array, without shuffle, ignore
        b = [a[0] for a in ArrayFlow(
                [np.arange(12)], 5, skip_incomplete=True)]
        self.assertEqual(2, len(b))
        np.testing.assert_array_equal(np.arange(0, 5), b[0])
        np.testing.assert_array_equal(np.arange(5, 10), b[1])

        # test dual arrays, without shuffle, no ignore
        b = list(ArrayFlow([np.arange(6), np.arange(12).reshape([6, 2])],
                                5))
        self.assertEqual(2, len(b))
        np.testing.assert_array_equal(np.arange(0, 5), b[0][0])
        np.testing.assert_array_equal(np.arange(5, 6), b[1][0])
        np.testing.assert_array_equal(np.arange(0, 10).reshape([5, 2]), b[0][1])
        np.testing.assert_array_equal(
            np.arange(10, 12).reshape([1, 2]), b[1][1])

        # test single array, with shuffle, no ignore
        b = [a[0] for a in ArrayFlow([np.arange(12)], 5, shuffle=True)]
        self.assertEqual(3, len(b))
        np.testing.assert_array_equal(np.arange(12), sorted(np.concatenate(b)))
