import unittest

import numpy as np
import pytest

from tfsnippet.dataflow import DataFlow


class MapperFlowTestCase(unittest.TestCase):

    def test_map_to_tuple(self):
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10)], batch_size=4)
        df = source.map(lambda x, y: (x + y,))
        self.assertIs(df.source, source)

        b = list(df)
        self.assertEquals(2, len(b))
        self.assertEquals(1, len(b[0]))
        np.testing.assert_array_equal([5, 7, 9, 11], b[0][0])
        np.testing.assert_array_equal([13], b[1][0])

    def test_map_to_list(self):
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10)], batch_size=4)
        df = source.map(lambda x, y: [x + y])
        self.assertIs(df.source, source)

        b = list(df)
        self.assertEquals(2, len(b))
        self.assertEquals(1, len(b[0]))
        np.testing.assert_array_equal([5, 7, 9, 11], b[0][0])
        np.testing.assert_array_equal([13], b[1][0])

    def test_errors(self):
        source = DataFlow.arrays([np.arange(5), np.arange(5, 10)], batch_size=4)
        df = source.map(lambda x, y: x + y)
        with pytest.raises(
                TypeError, match='The output of the ``mapper`` is expected to '
                                 'be a tuple or a list, but got a'):
            _ = list(df)

    def test_select(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        z = np.arange(10, 15)
        flow = DataFlow.arrays([x, y, z], batch_size=5).select([0, 2, 0])
        self.assertEquals(1, len(list(flow)))
        for b in flow:
            np.testing.assert_equal([x, z, x], b)


if __name__ == '__main__':
    unittest.main()
