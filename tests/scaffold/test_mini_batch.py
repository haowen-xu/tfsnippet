import unittest

import numpy as np
import pytest

from tfsnippet.scaffold import (minibatch_slices_iterator, ArrayMiniBatch,
                                MiniBatch)


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


class MiniBatchTestCase(unittest.TestCase):

    def test_get_iterator_reentrant(self):
        class MyMiniBatch(MiniBatch):
            def _get_iterator(self):
                return iter([123])

        m = MyMiniBatch()
        for b in m:
            self.assertEquals(123, b)
        for b in m:
            self.assertEquals(123, b)
            with pytest.raises(
                    RuntimeError,
                    match='get_iterator of MiniBatch is not re-entrant'):
                for b2 in m:
                    pass


class ArrayMiniBatchTestCase(unittest.TestCase):

    def test_property(self):
        m = ArrayMiniBatch(
            arrays=[np.arange(12).reshape([4, 3]), np.arange(4)],
            batch_size=5,
            shuffle=True,
            ignore_incomplete_batch=True
        )
        self.assertEquals(2, m.array_count)
        self.assertEquals(4, m.data_length)
        self.assertEquals(((3,), ()), m.data_shapes)
        self.assertEquals(5, m.batch_size)
        self.assertTrue(m.ignore_incomplete_batch)
        self.assertTrue(m.is_shuffled)

        # test default options
        m = ArrayMiniBatch([np.arange(12)], 5)
        self.assertFalse(m.ignore_incomplete_batch)
        self.assertFalse(m.is_shuffled)

    def test_errors(self):
        with pytest.raises(
                ValueError, match='`arrays` must not be empty'):
            _ = ArrayMiniBatch([], 3)
        with pytest.raises(
                ValueError, match='`arrays` must be numpy-like arrays'):
            _ = ArrayMiniBatch([np.arange(3).tolist()], 3)
        with pytest.raises(
                ValueError, match='`arrays` must be at least 1-d arrays'):
            _ = ArrayMiniBatch([np.array(0)], 3)
        with pytest.raises(
                ValueError, match='`arrays` must have the same data length'):
            _ = ArrayMiniBatch([np.arange(3), np.arange(4)], 3)

    def test_get_iterator(self):
        # test single array, without shuffle, no ignore
        b = [a[0] for a in ArrayMiniBatch([np.arange(12)], 5)]
        self.assertEquals(3, len(b))
        np.testing.assert_array_equal(np.arange(0, 5), b[0])
        np.testing.assert_array_equal(np.arange(5, 10), b[1])
        np.testing.assert_array_equal(np.arange(10, 12), b[2])

        # test single array, without shuffle, ignore
        b = [a[0] for a in ArrayMiniBatch(
                [np.arange(12)], 5, ignore_incomplete_batch=True)]
        self.assertEquals(2, len(b))
        np.testing.assert_array_equal(np.arange(0, 5), b[0])
        np.testing.assert_array_equal(np.arange(5, 10), b[1])

        # test dual arrays, without shuffle, no ignore
        b = list(ArrayMiniBatch([np.arange(6), np.arange(12).reshape([6, 2])],
                                5))
        self.assertEquals(2, len(b))
        np.testing.assert_array_equal(np.arange(0, 5), b[0][0])
        np.testing.assert_array_equal(np.arange(5, 6), b[1][0])
        np.testing.assert_array_equal(np.arange(0, 10).reshape([5, 2]), b[0][1])
        np.testing.assert_array_equal(
            np.arange(10, 12).reshape([1, 2]), b[1][1])

        # test single array, with shuffle, no ignore
        b = [a[0] for a in ArrayMiniBatch([np.arange(12)], 5, shuffle=True)]
        self.assertEquals(3, len(b))
        np.testing.assert_array_equal(np.arange(12), sorted(np.concatenate(b)))


if __name__ == '__main__':
    unittest.main()
