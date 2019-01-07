import unittest

import numpy as np
import pytest

from tfsnippet.dataflows import DataFlow


class MapperFlowTestCase(unittest.TestCase):

    def test_map_to_tuple(self):
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10)], batch_size=4)
        df = source.map(lambda x, y: (x + y,))
        self.assertIs(df.source, source)

        b = list(df)
        self.assertEqual(2, len(b))
        self.assertEqual(1, len(b[0]))
        np.testing.assert_array_equal([5, 7, 9, 11], b[0][0])
        np.testing.assert_array_equal([13], b[1][0])

    def test_map_to_list(self):
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10)], batch_size=4)
        df = source.map(lambda x, y: [x + y])
        self.assertIs(df.source, source)

        b = list(df)
        self.assertEqual(2, len(b))
        self.assertEqual(1, len(b[0]))
        np.testing.assert_array_equal([5, 7, 9, 11], b[0][0])
        np.testing.assert_array_equal([13], b[1][0])

    def test_map_array_indices(self):
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10),
                                  np.arange(10, 15), np.arange(15, 20)],
                                 batch_size=4)

        # assert map a single
        df = source.map(lambda x: [x + 1], array_indices=0)
        self.assertEqual(df.array_indices, (0,))
        b = list(df)
        np.testing.assert_array_equal(
            b[0],
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [10, 11, 12, 13],
                [15, 16, 17, 18],
            ]
        )
        np.testing.assert_array_equal(
            b[1],
            [[5], [9], [14], [19]]
        )

        # assert map multiples
        df = source.map(lambda x, y: [2 * x, 3 * y], array_indices=[1, 3])
        self.assertEqual(df.array_indices, (1, 3))
        b = list(df)
        np.testing.assert_array_equal(
            b[0],
            [
                [0, 1, 2, 3],
                [10, 12, 14, 16],
                [10, 11, 12, 13],
                [45, 48, 51, 54],
            ]
        )
        np.testing.assert_array_equal(
            b[1],
            [[4], [18], [14], [57]]
        )

    def test_errors(self):
        # test type error
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10)], batch_size=4)
        df = source.map(lambda x, y: x + y)
        with pytest.raises(
                TypeError, match='The output of the mapper is expected to '
                                 'be a tuple or a list, but got a'):
            _ = list(df)

        # test len(outputs) != len(inputs)
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10)],
                                 batch_size=4)
        df = source.map(lambda x, y: [x + y], [0, 1])
        with pytest.raises(
                ValueError, match='The number of output arrays of the mapper '
                                  'is required to match the inputs, since '
                                  '`array_indices` is specified: outputs 1 != '
                                  'inputs 2'):
            _ = list(df)

    def test_select(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        z = np.arange(10, 15)
        flow = DataFlow.arrays([x, y, z], batch_size=5).select([0, 2, 0])
        self.assertEqual(1, len(list(flow)))
        for b in flow:
            np.testing.assert_equal([x, z, x], b)
