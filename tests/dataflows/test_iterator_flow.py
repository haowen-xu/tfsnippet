import unittest

import numpy as np

from tfsnippet.dataflows import DataFlow


class IteratorFactoryFlowTestCase(unittest.TestCase):

    def test_iterator_factory(self):
        x_flow = DataFlow.arrays([np.arange(5)], batch_size=3)
        y_flow = DataFlow.arrays([np.arange(5, 10)], batch_size=3)
        flow = DataFlow.iterator_factory(lambda: (
            (x, y) for (x,), (y,) in zip(x_flow, y_flow)
        ))

        b = list(flow)
        self.assertEqual(2, len(b))
        self.assertEqual(2, len(b[0]))
        np.testing.assert_array_equal([0, 1, 2], b[0][0])
        np.testing.assert_array_equal([5, 6, 7], b[0][1])
        np.testing.assert_array_equal([3, 4], b[1][0])
        np.testing.assert_array_equal([8, 9], b[1][1])
