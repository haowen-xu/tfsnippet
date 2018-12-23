import unittest

import numpy as np
import pytest

from tfsnippet.utils import StatisticsCollector


class StatisticsCollectorTestCase(unittest.TestCase):

    def test_empty_scalar(self):
        collector = StatisticsCollector()
        self.assertEqual(collector.shape, ())
        self.assertFalse(collector.has_value)
        self.assertEqual(collector.counter, 0)
        self.assertAlmostEqual(collector.mean, 0.)
        self.assertAlmostEqual(collector.square, 0.)
        self.assertEqual(collector.weight_sum, 0.)

    def test_empty_vector(self):
        collector = StatisticsCollector(shape=(3, 2))
        self.assertEqual(collector.shape, (3, 2))
        self.assertFalse(collector.has_value)
        self.assertEqual(collector.counter, 0)
        np.testing.assert_almost_equal(collector.mean, np.zeros([3, 2]))
        np.testing.assert_almost_equal(collector.square, np.zeros([3, 2]))
        self.assertEqual(collector.weight_sum, 0.)

    def test_scalar_collect(self):
        collector = StatisticsCollector()
        collector.collect([2, 1, 7, 6])
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.mean, 4.)
        self.assertAlmostEqual(collector.square, 22.5)
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_var_std(self):
        collector = StatisticsCollector()
        collector.collect([2, 1, 7, 6])
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.var, 6.5)
        self.assertAlmostEqual(collector.stddev, 2.549509756796)

    def test_scalar_multi_collect(self):
        collector = StatisticsCollector()
        collector.collect(2)
        collector.collect(1, weight=3)
        collector.collect(7, weight=6)
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 3)
        self.assertAlmostEqual(collector.mean, 4.7)
        self.assertAlmostEqual(collector.square, 30.1)
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_reset(self):
        collector = StatisticsCollector()
        collector.collect([2, 1, 7, 6])
        collector.reset()
        self.assertFalse(collector.has_value)
        self.assertEqual(collector.counter, 0)
        self.assertAlmostEqual(collector.mean, 0.)
        self.assertAlmostEqual(collector.square, 0.)
        self.assertAlmostEqual(collector.var, 0.)
        self.assertAlmostEqual(collector.stddev, 0.)
        self.assertAlmostEqual(collector.weight_sum, 0.)

    def test_scalar_collect_batch(self):
        collector = StatisticsCollector()
        collector.collect([2, 1, 7], weight=[1, 3, 6])
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 3)
        self.assertAlmostEqual(collector.mean, 4.7)
        self.assertAlmostEqual(collector.square, 30.1)
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_scalar_collect_batch_weight_broadcast(self):
        collector = StatisticsCollector()
        collector.collect([2, 1, 7, 6], weight=1.)
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.mean, 4.)
        self.assertAlmostEqual(collector.square, 22.5)
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_vector_collect(self):
        collector = StatisticsCollector(shape=(3, 2))
        arr = np.arange(6).reshape([3, 2])
        collector.collect(arr)
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 1)
        np.testing.assert_almost_equal(collector.mean, arr)
        np.testing.assert_almost_equal(collector.square, arr ** 2)
        self.assertAlmostEqual(collector.weight_sum, 1.)

    def test_vector_multi_collect(self):
        collector = StatisticsCollector(shape=(3, 2))
        arr = np.arange(24).reshape([4, 3, 2])
        collector.collect(arr[0])
        collector.collect(arr[1], weight=2)
        collector.collect(arr[2], weight=3)
        collector.collect(arr[3], weight=4)
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 4)
        np.testing.assert_almost_equal(
            collector.mean,
            [[12, 13], [14, 15], [16, 17]]
        )
        np.testing.assert_almost_equal(
            collector.square,
            [[180., 205.], [232., 261.], [292., 325.]]
        )
        np.testing.assert_almost_equal(
            collector.var,
            np.maximum(collector.square - collector.mean ** 2, 0.)
        )
        np.testing.assert_almost_equal(
            collector.stddev,
            np.sqrt(np.maximum(collector.square - collector.mean ** 2, 0.))
        )
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_vector_collect_batch(self):
        collector = StatisticsCollector(shape=(3, 2))
        arr = np.arange(24).reshape([4, 3, 2])
        collector.collect(arr, weight=[1, 2, 3, 4])
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 4)
        np.testing.assert_almost_equal(
            collector.mean,
            [[12, 13], [14, 15], [16, 17]]
        )
        np.testing.assert_almost_equal(
            collector.square,
            [[180., 205.], [232., 261.], [292., 325.]]
        )
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_vector_collect_batch_weight_broadcast(self):
        collector = StatisticsCollector(shape=(3, 2))
        arr = np.arange(24).reshape([4, 3, 2])
        collector.collect(arr, weight=1.)
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 4)
        np.testing.assert_almost_equal(
            collector.mean,
            [[9, 10], [11, 12], [13, 14]]
        )
        np.testing.assert_almost_equal(
            collector.square,
            [[126., 145.], [166., 189.], [214., 241.]]
        )
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_collect_empty(self):
        collector = StatisticsCollector()
        collector.collect([])
        self.assertEqual(collector.shape, ())
        self.assertFalse(collector.has_value)
        self.assertEqual(collector.counter, 0)
        self.assertAlmostEqual(collector.mean, 0.)
        self.assertAlmostEqual(collector.square, 0.)
        self.assertEqual(collector.weight_sum, 0.)

    def test_collect_empty_weight(self):
        collector = StatisticsCollector()
        collector.collect([2, 1, 7, 6], weight=[])
        self.assertTrue(collector.has_value)
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.mean, 4.)
        self.assertAlmostEqual(collector.square, 22.5)
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_shape_mismatch(self):
        collector = StatisticsCollector(shape=(3, 2))
        with pytest.raises(
                ValueError,
                match=r'Shape mismatch: \(3,\) not ending with \(3, 2\)'):
            collector.collect([1, 2, 3])
