import unittest

import numpy as np
import pytest
from mock import Mock

from tfsnippet.dataflows import DataMapper, SlidingWindow


class DataMapperTestCase(unittest.TestCase):

    def test_error(self):
        dm = DataMapper()
        dm._transform = Mock(return_value=np.array([1, 2, 3]))
        with pytest.raises(TypeError, match='The output of .* is neither '
                                            'a tuple, nor a list'):
            dm(np.array([1, 2, 3]))


class SlidingWindowTestCase(unittest.TestCase):

    def test_props(self):
        arr = np.arange(13)
        sw = SlidingWindow(arr, window_size=3)
        self.assertIs(arr, sw.data_array)
        self.assertEqual(3, sw.window_size)

    def test_transform(self):
        arr = np.arange(13)
        sw = SlidingWindow(arr, window_size=3)
        np.testing.assert_equal(
            [[0, 1, 2], [5, 6, 7], [3, 4, 5]],
            sw(np.asarray([0, 5, 3]))[0]
        )

    def test_as_flow(self):
        arr = np.arange(13)
        sw = SlidingWindow(arr, window_size=3)
        batches = list(sw.as_flow(batch_size=4))
        self.assertEqual(3, len(batches))
        np.testing.assert_equal(
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
            batches[0][0]
        )
        np.testing.assert_equal(
            [[4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
            batches[1][0]
        )
        np.testing.assert_equal(
            [[8, 9, 10], [9, 10, 11], [10, 11, 12]],
            batches[2][0]
        )
