import unittest

import numpy as np
import pytest

from tfsnippet.utils import *


class MiniBatchSlicesIteratorTestCase(unittest.TestCase):

    def test_minibatch_slices_iterator(self):
        self.assertEqual(
            list(minibatch_slices_iterator(0, 10, False)),
            []
        )
        self.assertEqual(
            list(minibatch_slices_iterator(9, 10, False)),
            [slice(0, 9, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 10, False)),
            [slice(0, 10, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 9, False)),
            [slice(0, 9, 1), slice(9, 10, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 9, True)),
            [slice(0, 9, 1)]
        )


class SplitNumpyArraysTestCase(unittest.TestCase):

    def test_error_inputs(self):
        # test error inputs
        with pytest.raises(
                ValueError, match='At least one of `portion` and `size` should '
                                  'be specified.'):
            split_numpy_arrays([])

        with pytest.raises(
                ValueError, match='At least one of `portion` and `size` should '
                                  'be specified.'):
            split_numpy_arrays([np.arange(1)])

        with pytest.raises(
                ValueError, match='The length of specified arrays are not '
                                  'equal.'):
            split_numpy_arrays([np.arange(10), np.arange(11)], portion=0.2)

        with pytest.raises(
                ValueError, match='`portion` must range from 0.0 to 1.0.'):
            split_numpy_arrays([np.arange(10)], portion=-0.1)

        with pytest.raises(
                ValueError, match='`portion` must range from 0.0 to 1.0.'):
            split_numpy_arrays([np.arange(10)], portion=1.1)

    def test_empty_inputs(self):
        self.assertEqual(split_numpy_arrays([], portion=0.2), ((), ()))
        self.assertEqual(split_numpy_arrays([], size=10), ((), ()))

    def test_split_by_size(self):
        left, right = split_numpy_arrays(
            [np.arange(10)], size=-1, shuffle=False)
        np.testing.assert_equal(left, [np.arange(10)])
        np.testing.assert_equal(right, [[]])

        left, right = split_numpy_arrays(
            [np.arange(10)], size=0, shuffle=False)
        np.testing.assert_equal(left, [np.arange(10)])
        np.testing.assert_equal(right, [[]])

        left, right = split_numpy_arrays(
            [np.arange(10)], size=1, shuffle=False)
        np.testing.assert_equal(left, [np.arange(9)])
        np.testing.assert_equal(right, [[9]])

        left, right = split_numpy_arrays(
            [np.arange(10)], size=9, shuffle=False)
        np.testing.assert_equal(left, [[0]])
        np.testing.assert_equal(right, [np.arange(1, 10)])

        left, right = split_numpy_arrays(
            [np.arange(10)], size=10, shuffle=False)
        np.testing.assert_equal(left, [[]])
        np.testing.assert_equal(right, [np.arange(10)])

        left, right = split_numpy_arrays(
            [np.arange(10)], size=11, shuffle=False)
        np.testing.assert_equal(left, [[]])
        np.testing.assert_equal(right, [np.arange(10)])

    def test_shuffling_with_multiple_arrays(self):
        left, right = split_numpy_arrays(
            [np.arange(10), np.arange(10, 20)], size=1, shuffle=True)
        self.assertEqual(len(left[0]), 9)
        self.assertEqual(len(left[1]), 9)
        self.assertEqual(len(right[0]), 1)
        self.assertEqual(len(right[1]), 1)
        self.assertEqual(
            set(list(left[0]) + list(right[0])), set(np.arange(10)))
        self.assertEqual(
            set(list(left[1]) + list(right[1])), set(np.arange(10, 20)))
        np.testing.assert_equal(left[0] + 10, left[1])
        np.testing.assert_equal(right[0] + 10, right[1])

    def test_split_multi_dimensional_data(self):
        left, right = split_numpy_arrays(
            [np.arange(24).reshape([6, 2, 2])], size=3, shuffle=False)
        np.testing.assert_equal(left, [np.arange(12).reshape([3, 2, 2])])
        np.testing.assert_equal(right, [np.arange(12, 24).reshape([3, 2, 2])])

    def test_split_by_portion(self):
        left, right = split_numpy_arrays(
            [np.arange(10)], portion=0.1, shuffle=False)
        np.testing.assert_equal(left, [np.arange(9)])
        np.testing.assert_equal(right, [[9]])
        left, right = split_numpy_arrays(
            [np.arange(10)], portion=0.9, shuffle=False)
        np.testing.assert_equal(left, [[0]])
        np.testing.assert_equal(right, [np.arange(1, 10)])

    def test_split_numpy_array(self):
        left, right = split_numpy_array(
            np.arange(10), portion=0.1, shuffle=False)
        np.testing.assert_equal(left, np.arange(9))
        np.testing.assert_equal(right, [9])
