import unittest

import numpy as np

from tfsnippet.dataflow import DataFlow


class MapperFlowTestCase(unittest.TestCase):

    def test_map(self):
        df = DataFlow.from_arrays([np.arange(5), np.arange(5, 10)],
                                  batch_size=4)
        df = df.map(lambda arr: (arr[0] + arr[1],))

        b = list(df)
        self.assertEquals(2, len(b))
        self.assertEquals(1, len(b[0]))
        np.testing.assert_array_equal([5, 7, 9, 11], b[0][0])
        np.testing.assert_array_equal([13], b[1][0])


if __name__ == '__main__':
    unittest.main()
