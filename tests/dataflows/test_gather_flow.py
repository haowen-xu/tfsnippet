import unittest

import numpy as np
import pytest

from tfsnippet.dataflows import DataFlow
from tfsnippet.dataflows.gather_flow import GatherFlow


class GatherFlowTestCase(unittest.TestCase):

    def test_flow(self):
        x_flow = DataFlow.arrays([np.arange(10)], batch_size=4)
        y_flow = DataFlow.arrays([np.arange(10, 17)], batch_size=4)
        flow = DataFlow.gather([x_flow, y_flow])
        self.assertIsInstance(flow, GatherFlow)
        self.assertEqual((x_flow, y_flow), flow.flows)
        batches = list(flow)
        self.assertEqual(2, len(batches))
        np.testing.assert_equal(np.arange(4), batches[0][0])
        np.testing.assert_equal(np.arange(10, 14), batches[0][1])
        np.testing.assert_equal(np.arange(4, 8), batches[1][0])
        np.testing.assert_equal(np.arange(14, 17), batches[1][1])

    def test_errors(self):
        with pytest.raises(
                ValueError, match='At least one flow must be specified'):
            _ = DataFlow.gather([])
        with pytest.raises(TypeError, match='Not a DataFlow'):
            _ = DataFlow.gather([1])
